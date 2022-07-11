from .features import Features


def polynomial_kernel(x, y):
    d = x.shape[-1]
    dot = x @ y.transpose(-2, -1)
    return (dot / d + 1) ** 3


def kid(x, y, kernel=polynomial_kernel):
    m = x.shape[-2]
    n = y.shape[-2]
    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    kxx_sum = kxx.sum([-1, -2]) - kxx.diagonal(dim1=-1, dim2=-2).sum(-1)
    kyy_sum = kyy.sum([-1, -2]) - kyy.diagonal(dim1=-1, dim2=-2).sum(-1)
    kxy_sum = kxy.sum([-1, -2])
    term_1 = kxx_sum / m / (m - 1)
    term_2 = kyy_sum / n / (n - 1)
    term_3 = kxy_sum * 2 / m / n
    return term_1 + term_2 - term_3


def compute_kid(a: Features, b: Features) -> float:
    return kid(a.features, b.features)


def test_kid_directories():
    import numpy as np
    from cleanfid import fid

    np.random.seed(123)
    reference_kid_score = fid.compute_kid(
        "tests/pixelart/dataset_a", "tests/pixelart/dataset_b"
    )
    np.random.seed(123)
    features_a = Features.from_directory("tests/pixelart/dataset_a")
    features_b = Features.from_directory("tests/pixelart/dataset_b")
    assert compute_kid(features_a, features_b) == reference_kid_score
