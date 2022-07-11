from PIL import Image
import torch
import torchvision.transforms.functional as TF
from resize_right import resize

from . import settings


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        print("fail", image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return resize(
            TF.to_tensor(Image.open(self.image_paths[index]).convert("RGB")),
            out_shape=settings.RESIZE_SHAPE,
        )
