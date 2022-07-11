from typing import Tuple
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from resize_right import resize as resize_right

from . import settings


# only passes atol=1-3
# def resize(image: torch.Tensor, output_size: Tuple[int, int] = settings.RESIZE_SHAPE) -> torch.Tensor:
#     return resize_right(image, out_shape=output_size, pad_mode="reflect").clamp(0, 1)


def resize(
    image: torch.Tensor, output_size: Tuple[int, int] = settings.RESIZE_SHAPE
) -> torch.Tensor:
    return (
        torch.from_numpy(
            cleanfid_resize(image.mul(255).permute(1, 2, 0).cpu().numpy(), output_size)
        )
        .permute(2, 0, 1)
        .div(255)
        .to(image.device)
    )


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


def test_resize_same():
    from PIL import Image
    from cleanfid.resize import build_resizer

    image = Image.open("tests/pixelart/dataset_a/out_00003.png")
    reference_resize = build_resizer("clean")

    resized = resize(TF.to_tensor(image))
    assert np.allclose(
        reference_resize(np.array(image)),
        resized.permute(1, 2, 0).mul(255).numpy(),
        atol=1e-7,
    )
