import torch
import torch.nn as nn
import contextlib

from .check_download_inception import check_download_inception


@contextlib.contextmanager
def disable_gpu_fuser_on_pt19():
    # On PyTorch 1.9 a CUDA fuser bug prevents the Inception JIT model to run. See
    #   https://github.com/GaParmar/clean-fid/issues/5
    #   https://github.com/pytorch/pytorch/issues/64062
    if torch.__version__.startswith("1.9."):
        old_val = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_gpu(False)
    yield
    if torch.__version__.startswith("1.9."):
        torch._C._jit_override_can_fuse_on_gpu(old_val)


class InceptionV3W(nn.Module):
    def __init__(self, path=None):
        """
        Wrapper around Inception V3 torchscript model provided here
        https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt

        Args:
            path: locally saved inception weights
        """
        super().__init__()
        # download the network if it is not present at the given directory
        # use the current directory by default
        inception_path = check_download_inception(fpath=path)
        self.base = torch.jit.load(inception_path).eval()
        self.layers = self.base.layers

    def forward(self, x):
        """
        Get the inception features without resizing

        Args:
            x: Image with values in range [0,255]
        """
        with disable_gpu_fuser_on_pt19():
            bs = x.shape[0]

            # make sure it is resized already
            assert x.shape[-2:] == (299, 299)
            # apply normalization
            x1 = x - 128
            x2 = x1 / 128
            features = self.layers.forward(
                x2,
            ).view((bs, 2048))

            return features
