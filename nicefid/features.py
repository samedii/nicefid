from typing import Union, Callable
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

from cleanfid import fid

MODE = "clean"


class Features:
    features: np.array

    def __init__(self, features: np.array):
        self.features = features

    @property
    def mu(self):
        return np.mean(self.features, axis=0)

    @property
    def sigma(self):
        return np.cov(self.features, rowvar=False)

    @staticmethod
    def from_directory(
        path: Union[str, Path],
        batch_size=32,
        n_workers=12,
        device=torch.device("cuda"),
    ) -> "Features":
        feature_model = fid.build_feature_extractor(MODE, device)
        features = fid.get_folder_features(
            str(path),
            feature_model,
            num_workers=n_workers,
            batch_size=batch_size,
            device=torch.device("cuda"),
            mode=MODE,
            description=f"Extracting features for {Path(path).name} : ",
            verbose=True,
            custom_image_tranform=None,
        )
        return Features(features=features)

    @staticmethod
    def from_generator(
        generator: Callable[[], torch.Tensor], device=torch.device("cuda")
    ) -> "Features":
        """
        Args:
            generator: A generator that yields batches of images NCHW in the
                range between 0 and 1.
            device: The device to use for the feature extractor.
        """
        model = fid.build_feature_extractor(MODE, device)
        fn_resize = fid.build_resizer(MODE)

        features = list()
        for images in generator():
            if images.ndim != 4:
                raise ValueError(f"Expected NCHW images but got {images.shape}.")

            resized_images = list()
            for current_image in images:
                image_numpy = current_image.mul(255).cpu().numpy().transpose((1, 2, 0))
                resized_image = fn_resize(image_numpy)
                resized_images.append(torch.tensor(resized_image.transpose((2, 0, 1))))
            batch_features = fid.get_batch_features(
                torch.stack(resized_images), model, device
            )
            features.append(batch_features)
        features = np.concatenate(features)
        return Features(features=features)

    @staticmethod
    def from_name(name: str) -> "Features":
        raise NotImplementedError()

    def save(self, name: str):
        raise NotImplementedError()


def test_folder_and_generator_equal():
    from PIL import Image
    import torchvision.transforms.functional as TF

    directory = Path("tests/pixelart/dataset_a")
    a = Features.from_directory(directory, batch_size=1)

    def generator():
        for image_path in directory.glob("*.png"):
            yield TF.to_tensor(Image.open(image_path))[None]

    b = Features.from_generator(generator)

    assert np.allclose(a.features.mean(axis=0), b.features.mean(axis=0), atol=1e-3)
