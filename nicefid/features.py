from typing import Union
from pathlib import Path
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
    ):
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
