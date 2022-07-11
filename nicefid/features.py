from typing import Iterator, Union, Iterator
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

from .inception_v3w import InceptionV3W
from .list_images import list_images
from .image_dataset import ImageDataset
from .resize import resize
from . import settings


class Features:
    features: torch.Tensor

    def __init__(self, features: np.array):
        self.features = features

    @staticmethod
    def from_directory(
        path: Union[str, Path],
        batch_size=32,
        n_workers=12,
        device=torch.device("cuda"),
        model=None,
    ) -> "Features":
        image_dataset = ImageDataset(list_images(path))
        data_loader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=n_workers,
        )
        return Features.from_iterator(data_loader, device, model)

    @staticmethod
    def from_iterator(
        iterator: Iterator[torch.Tensor],
        device=torch.device("cuda"),
        model=None,
    ) -> "Features":
        """
        Args:
            iterator: An iterator that yields batches of images NCHW in the
                range between 0 and 1.
            device: The device to use for the feature extractor.
        """
        if model is None:
            model = InceptionV3W().eval().requires_grad_(False).to(device)

        features = list()
        for images in iterator:
            if images.ndim != 4:
                raise ValueError(f"Expected NCHW images but got {images.shape}")
            if images.shape[1] != 3:
                raise ValueError(f"Expected 3 channels (RGB) but got {images.shape}")
            if images.max() >= 128:
                raise ValueError(
                    f"Expected image in range [0, 1] but got far larger max value {images.max().item()}"
                )

            images = images.to(device)
            if images.shape[-2:] != settings.RESIZE_SHAPE:
                images = torch.stack([resize(image) for image in images])

            batch_features = model(images.mul(255)).cpu()
            features.append(batch_features)
        features = torch.cat(features)
        return Features(features=features)

    @staticmethod
    def from_path(path: Union[str, Path]) -> "Features":
        return Features(features=torch.load(path))

    def save(self, path: Union[str, Path]):
        torch.save(self.features, path)


def test_features_against_reference():
    from cleanfid import fid

    directory = "tests/pixelart/dataset_a"
    reference_features = fid.get_folder_features(directory)
    reimplementation = Features.from_directory(directory)
    assert np.allclose(
        reference_features.mean(axis=0),
        reimplementation.features.mean(axis=0),
        atol=1e-3,
    )


def test_folder_and_iterator_equal():
    from PIL import Image
    import torchvision.transforms.functional as TF

    directory = Path("tests/pixelart/dataset_a")
    a = Features.from_directory(directory, batch_size=1)

    def iterator():
        for image_path in directory.glob("*.png"):
            yield TF.to_tensor(Image.open(image_path))[None]

    b = Features.from_iterator(iterator())

    assert np.allclose(a.features.mean(axis=0), b.features.mean(axis=0), atol=1e-3)


def test_save_load_works():
    a = Features.from_directory("tests/pixelart/dataset_a")
    path = Path("test_features.pt")
    a.save(path)
    b = Features.from_path(path)
    path.unlink()
    assert np.allclose(a.features, b.features)
