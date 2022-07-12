import numpy as np
from scipy import linalg
import torch

from .features import Features
from .sqrtm_eig import sqrtm_eig


# FID 307 vs 310 compared to cleanfid
# def fid(x, y, eps=1e-6):
#     x_mean = x.mean(dim=0)
#     y_mean = y.mean(dim=0)
#     mean_term = (x_mean - y_mean).pow(2).sum()
#     x_cov = torch.cov(x.T)
#     y_cov = torch.cov(y.T)
#     eps_eye = torch.eye(x_cov.shape[0], device=x_cov.device, dtype=x_cov.dtype) * eps
#     x_cov_sqrt = sqrtm_eig(x_cov + eps_eye)
#     cov_term = torch.trace(
#         x_cov + y_cov - 2 * sqrtm_eig(x_cov_sqrt @ y_cov @ x_cov_sqrt + eps_eye)
#     )
#     return mean_term + cov_term


def cleanfid_fid(x: np.array, y: np.array, eps=1e-6) -> float:
    mu1 = np.mean(x, axis=0)
    sigma1 = np.cov(x, rowvar=False)
    mu2 = np.mean(y, axis=0)
    sigma2 = np.cov(y, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError("Training and test mean vectors have different lengths")
    if sigma1.shape != sigma2.shape:
        raise ValueError("Training and test covariances have different dimensions")

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_fid(a: Features, b: Features) -> float:
    # return fid(a.features, b.features)
    return cleanfid_fid(a.features.cpu().numpy(), b.features.cpu().numpy())


def test_fid_directories():
    from cleanfid import fid

    reference_fid_score = fid.compute_fid(
        "tests/pixelart/dataset_a", "tests/pixelart/dataset_b"
    )

    features_a = Features.from_directory("tests/pixelart/dataset_a")
    features_b = Features.from_directory("tests/pixelart/dataset_b")
    new_fid = compute_fid(features_a, features_b)
    print(reference_fid_score, new_fid)
    assert np.allclose(new_fid, reference_fid_score, rtol=1e-3)


def test_fid_random_noise(tmpdir):
    from pathlib import Path
    from PIL import Image
    from cleanfid import fid

    temporary_directory = Path(tmpdir)
    dataset_a_directory = temporary_directory / "dataset_a"
    dataset_a_directory.mkdir()
    for index in range(128):
        Image.fromarray(torch.rand((32, 32, 3)).mul(255).byte().numpy()).save(
            dataset_a_directory / f"{index}.bmp"
        )

    dataset_b_directory = temporary_directory / "dataset_b"
    dataset_b_directory.mkdir()
    for index in range(128):
        Image.fromarray(torch.rand((32, 32, 3)).mul(255).byte().numpy()).save(
            dataset_b_directory / f"{index}.bmp"
        )

    reference_fid_score = fid.compute_fid(
        str(dataset_a_directory), str(dataset_b_directory)
    )

    features_a = Features.from_directory(dataset_a_directory)
    features_b = Features.from_directory(dataset_b_directory)
    new_fid = compute_fid(features_a, features_b)
    print(reference_fid_score, new_fid)
    assert np.allclose(new_fid, reference_fid_score, rtol=1e-3)
