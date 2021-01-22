import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,6,7"
import argparse
import logging
import json
import time
import glob

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit

from PIL import Image
import cv2
import torchvision

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from torch import nn
from torch.nn import DataParallel

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from camelyon16.data.wsi_producer_slide_window import WSIPatchDataset


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('mask_path', default=None, metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='0,1,2,3', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=4, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')
#max_batch_size = 32
#pytorch_time = 0
trt_time = 0
trt_infer_time = 0

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    '''
    allocate buffers from host and device for inputs, outputs tensor.
    inputs : [(host_memory, device_memory)]
    bindings : [device_memory]
    outputs : [(host_memory, device_memory)]
    '''

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        # print(binding) # 绑定的输入输出
        # print(engine.get_binding_shape(binding)) # get_binding_shape 是变量的大小
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size

        # volume 计算可迭代变量的空间，指元素个数
        # size = trt.volume(engine.get_binding_shape(binding)) # 如果采用固定bs的onnx，则采用该句
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # get_binding_dtype  获得binding的数据类型
        # nptype等价于numpy中的dtype，即数据类型
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)   # 创建锁业内存
        device_mem = cuda.mem_alloc(host_mem.nbytes)     # cuda分配空间

        # print(int(device_mem)) # binding在计算图中的缓冲地址
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        # Append to the appropriate list.
        # this function can determine where the device_memory comes from, input or output.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def get_engine(max_batch_size=128, onnx_file_path="", engine_file_path="",
               fp16_mode=True, int8_mode=False, save_engine=False,
               ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    """
       params max_batch_size:      预先指定大小好分配显存
       params onnx_file_path:      onnx文件路径
       params engine_file_path:    待保存的序列化的引擎文件路径
       params fp16_mode:           是否采用FP16
       params save_engine:         是否保存引擎
       returns:                    ICudaEngine
       """
    #通过加载onnx文件，构建engine.

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        #输入ONNX，生成TensorRT。
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network() as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30  # Your workspace size
            builder.max_batch_size = max_batch_size
            # print("builder.max_batch_size",builder.max_batch_size)
            # pdb.set_trace()
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False

            if int8_mode:
                # To be updated
                raise NotImplementedError

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            #保存engine。
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("build!")
        return build_engine(max_batch_size, save_engine)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=128):
    # Transfer data from CPU to the GPU. 将数据从CPU转移到GPU。
    t1 = time.time()
    # [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]
    t2 = time.time()
    
    # Run inference. 执行模型
    # context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    context.execute(batch_size=batch_size, bindings=bindings)
    # time.sleep(1)

    t3 = time.time()
    # Transfer predictions back from the GPU. 将预测结果从GPU取出。
    # [cuda.c(out.host, out.device, stream) for out in outputs]
    [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]

    t4 = time.time()
    global trt_time
    global trt_infer_time
    trt_time += (t4 - t1)
    trt_infer_time += (t3 - t2)
    print("trt sum time:", trt_time)
    print("trt infer sum time:", trt_infer_time)
    print("one time:", t4-t1)
    # print("TensorRT input time:",t2-t1)
    # print("Inference time with the TensorRT engine: {}".format(t3-t2))
    # print("TensorRT output time:" + str(t4 - t3))

    # Synchronize the stream 线程同步
    # stream.synchronize()
    # Return only the host outputs.

    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


# def chose_model(mod):
#     if mod == 'resnet50':
#         model = models.resnet50(pretrained=False)
#     else:
#         raise Exception("I have not add any models. ")
#     return model


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def get_probs_map(engine, buffers, dataloader, pytorch_model):
    probs_map = np.zeros((dataloader.dataset.X_idces, dataloader.dataset.Y_idces))    
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    #context = engine.create_execution_context()
    inputs, outputs, bindings, stream = buffers

    with engine.create_execution_context() as context:

        for (data, x_mask, y_mask) in dataloader:
            #print(data.dtype)

            # t1 = time.time()
            # pytorch_feat = pytorch_model(data)
            # t1_5 = time.time()
            # #print(pytorch_feat[0])
            # t2 = time.time()

            #global pytorch_time
            #pytorch_time += (t2 - t1)
            #print("pytorch sum time:", pytorch_time)

            #pytorch_feat = sigmoid(pytorch_feat.cpu().data.numpy())

            data = data.numpy().astype(dtype=np.float32)
            #shape_of_output = (1, 3, 256, 256)
        
            inputs[0].host = data.reshape(-1)

            t3 = time.time()
            trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t4 = time.time()

            trt_feat = sigmoid(trt_outputs[0])

            #mse = np.mean((pytorch_feat - trt_feat)**2)
            #print("Inference time with the TensorRT engine: {}".format(t4-t3))
            #print("Inference time with the PyTorch model: {}".format(t2-t1))
            #print("print time:",t2-t1_5)
            #print('MSE Error = {}'.format(mse))

            #feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)
        
            '''
            if len(output.shape) == 1:
                probs = output.sigmoid().cpu().data.numpy().flatten()
            else:
                probs = output[:,:].sigmoid().cpu().data.numpy().flatten()
            probs_map[x_mask, y_mask] = probs
            '''
            if (probs_map[x_mask, y_mask]).shape <(128,):
                probs_map[x_mask, y_mask] = probs_map[x_mask, y_mask]
            else:
                probs_map[x_mask, y_mask] = trt_feat

            #print("trt_outputs[0]:",trt_outputs[0])
            count += 1

            time_spent = time.time() - time_now
            time_now = time.time()
            print("Get probsmap time: {}".format(time_spent))
            logging.info(
                '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
                .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                    dataloader.dataset._rotate, count, num_batch, time_spent))

    return probs_map


def make_dataloader(args, cnn, flip='NONE', rotate='NONE'):
    
    num_GPU = len(args.GPU.split(','))
    #batch_size = cnn['batch_size'] * num_GPU
    num_workers = args.num_workers

    dataloader = DataLoader(
        WSIPatchDataset(args.wsi_path,
                        image_size=cnn['image_size'],
                        crop_size=cnn['crop_size'], normalize=True,
                        flip=flip, rotate=rotate),
        batch_size=128, num_workers=num_workers, drop_last=False)
   
    return dataloader


def run(args, engine, buffers, pytorch_model):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cnn_path) as f:
        cnn = json.load(f)
    paths = glob.glob(os.path.join(args.wsi_path, '*.svs'))
    for path in paths:
        print(path)
        args.wsi_path = path
        npy_name = os.path.basename(path)
        npy_path = os.path.join(args.probs_map_path, npy_name[:-4] + '_prob.npy')
        if os.path.exists(npy_path):
            continue
        mask_path = os.path.join(args.mask_path,npy_name[:-4] + '_tissue.npy')
        mask = np.load(mask_path)
        '''
        ckpt = torch.load(args.ckpt_path)
        model = chose_model(cnn['model'])
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 1)
        model.load_state_dict(ckpt['state_dict'])
        model = DataParallel(model, device_ids=None)
        model = model.cuda().eval()
        '''
        if not args.eight_avg:
            dataloader = make_dataloader(
                args, cnn, flip='NONE', rotate='NONE')
            probs_map = get_probs_map(engine, buffers, dataloader, pytorch_model)
        else:
            probs_map = np.zeros(mask.shape)

            starttime=time.time()
            dataloader = make_dataloader(
                args,mask_path, cnn, flip='NONE', rotate='NONE')
            probs_map += get_probs_map(engine, dataloader)

            dataloader = make_dataloader(
                args, mask_path,cnn, flip='NONE', rotate='ROTATE_90')
            probs_map += get_probs_map(engine, dataloader)

            dataloader = make_dataloader(
                args, mask_path,cnn, flip='NONE', rotate='ROTATE_180')
            probs_map += get_probs_map(engine, dataloader)

            dataloader = make_dataloader(
                args, mask_path,cnn, flip='NONE', rotate='ROTATE_270')
            probs_map += get_probs_map(engine, dataloader)

            dataloader = make_dataloader(
                args, mask_path,cnn, flip='FLIP_LEFT_RIGHT', rotate='NONE')
            probs_map += get_probs_map(engine, dataloader)

            dataloader = make_dataloader(
                args, mask_path,cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
            probs_map += get_probs_map(engine, dataloader)

            dataloader = make_dataloader(
                args, mask_path,cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
            probs_map += get_probs_map(engine, dataloader)

            dataloader = make_dataloader(
                args, mask_path,cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
            probs_map += get_probs_map(engine, dataloader)

            probs_map /= 8
            endtime=time.time()
            print("Data loader time: {}".format(endtime-starttime))
        np.save(npy_path, probs_map)



def main():
    max_batch_size = 128
    onnx_model_path = 'best.onnx'

    TRT_LOGGER = trt.Logger()  # This logger is required to build an engine
    # These two modes are dependent on hardwares
    fp16_mode = True
    int8_mode = False
    #trt_engine_path = './model_fp16_{}_int8_{}_batchs.trt'.format(fp16_mode, int8_mode)
    trt_engine_path = './model_fp16_True_int8_False.trt'

    args = parser.parse_args()
    engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode, save_engine=True)
    context = engine.create_execution_context()
    buffers = allocate_buffers(engine)

    pytorch_model = torchvision.models.resnet50(pretrained=False).cuda()
    weight_path = '/home/sdb/gongxiang/CAMELYON-master/CAMELYON-master/camelyon16/bin/ckptpath50/best.ckpt'
    ckpt = torch.load(weight_path)
    fc_features = pytorch_model.fc.in_features
    pytorch_model.fc = nn.Linear(fc_features, 1)
    pytorch_model.load_state_dict(ckpt['state_dict'])
    pytorch_model = DataParallel(pytorch_model, device_ids=None)
    # pytorch_model = pytorch_model.cuda().eval()

    run(args, engine, buffers, pytorch_model)
    #run(args, engine, buffers)

    del engine

    print('TensorRT ok')


if __name__ == '__main__':
    main()
