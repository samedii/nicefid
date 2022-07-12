from typing import Tuple
from PIL import Image
import numpy as np


def cleanfid_resize(image: np.array, output_size: Tuple[int, int]) -> np.array:
    s1, s2 = output_size

    def resize_single_channel(x_np):
        img = Image.fromarray(x_np.astype(np.float32), mode="F")
        img = img.resize(output_size, resample=Image.Resampling.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(s1, s2, 1)

    def func(x):
        x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
        x = np.concatenate(x, axis=2).astype(np.float32)
        return x

    return func(image)
