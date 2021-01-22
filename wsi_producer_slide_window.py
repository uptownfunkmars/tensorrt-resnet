import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
import math


class WSIPatchDataset(Dataset):

    def __init__(self, wsi_path, image_size=256, crop_size=224,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._wsi_path = wsi_path
        self._image_size = image_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self._slide = openslide.OpenSlide(self._wsi_path)
        self.X_slide, self.Y_slide = self._slide.level_dimensions[0]
        self.X_idces = math.ceil(self.X_slide / self._image_size)
        self.Y_idces = math.ceil(self.Y_slide / self._image_size)

    def __len__(self):
        return self.X_idces * self.Y_idces

    def __getitem__(self, coord_idx):
        x_idx = coord_idx // self.Y_idces
        y_idx = coord_idx % self.Y_idces
        x_mask = self._image_size * x_idx
        y_mask = self._image_size * y_idx
        x = x_mask if int(x_mask + self._image_size) < self.X_slide else self.X_slide - self._image_size
        y = y_mask if int(y_mask + self._image_size) < self.Y_slide else self.Y_slide - self._image_size

        img = self._slide.read_region(
            (x, y), 1, (self._image_size, self._image_size)).convert('RGB')


        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return (img, x_idx, y_idx)

# dataloader = DataLoader(
#     WSIPatchDataset(r'D:\publicData\TCGA\svs\TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs',
#                     image_size=768,
#                     crop_size=672, normalize=True),
#                     batch_size=128, drop_last=False)
# num_batch = len(dataloader)
# for (data, x_mask, y_mask) in dataloader:
#     print(x_mask,y_mask)