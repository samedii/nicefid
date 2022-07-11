import os
import platform
import urllib.request
import shutil

inception_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"


def check_download_inception(fpath=None):
    """
    Download the pretrined inception weights if it does not exists
    Args:
        fpath - output folder path
    """
    if fpath is None:
        fpath = "./" if platform.system() == "Windows" else "/tmp"
    inception_path = os.path.join(fpath, "inception-2015-12-05.pt")
    if not os.path.exists(inception_path):
        # download the file
        with urllib.request.urlopen(inception_url) as response, open(
            inception_path, "wb"
        ) as f:
            shutil.copyfileobj(response, f)
    return inception_path
