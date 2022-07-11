from cleanfid import fid

from .features import Features


def compute_fid(a: Features, b: Features) -> float:
    return fid.frechet_distance(a.mu, a.sigma, b.mu, b.sigma)


def test_fid_directories():
    reference_fid_score = fid.compute_fid(
        "tests/pixelart/dataset_a", "tests/pixelart/dataset_b"
    )

    features_a = Features.from_directory("tests/pixelart/dataset_a")
    features_b = Features.from_directory("tests/pixelart/dataset_b")
    assert compute_fid(features_a, features_b) == reference_fid_score
