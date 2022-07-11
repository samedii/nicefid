from cleanfid import fid

from .features import Features


def compute_kid(a: Features, b: Features) -> float:
    return fid.kernel_distance(a.features, b.features)


def test_kid_directories():
    import numpy as np

    np.random.seed(123)
    reference_kid_score = fid.compute_kid(
        "tests/pixelart/dataset_a", "tests/pixelart/dataset_b"
    )
    np.random.seed(123)
    features_a = Features.from_directory("tests/pixelart/dataset_a")
    features_b = Features.from_directory("tests/pixelart/dataset_b")
    assert compute_kid(features_a, features_b) == reference_kid_score
