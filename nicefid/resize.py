from typing import Tuple
import numpy as np
import torch
import torchvision.transforms.functional as TF
from resize_right import resize as resize_right

from . import settings
from .cleanfid_resize import cleanfid_resize


# def resize(
#     images: torch.Tensor, output_size: Tuple[int, int] = settings.RESIZE_SHAPE
# ) -> torch.Tensor:
#     """
#     Resize images to the given output size. Atol vs cleanfid implementation is around 1e-3.
#     """
#     return resize_right(
#         images,
#         out_shape=output_size,
#         pad_mode="reflect",
#     ).clamp(0, 1)


def resize(
    images: torch.Tensor, output_size: Tuple[int, int] = settings.RESIZE_SHAPE
) -> torch.Tensor:
    return torch.stack(
        [
            (
                torch.from_numpy(cleanfid_resize(image, output_size))
                .permute(2, 0, 1)
                .div(255)
                .to(images.device)
            )
            for image in images.mul(255).permute(0, 2, 3, 1).cpu().numpy()
        ]
    )


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
