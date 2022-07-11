from PIL import Image
import torch
import torchvision.transforms.functional as TF

from .resize import resize


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return resize(TF.to_tensor(image))
