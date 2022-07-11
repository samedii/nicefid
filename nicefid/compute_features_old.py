import math
import torch
from tqdm import trange


def compute_features(accelerator, sample_fn, extractor_fn, n, batch_size):
    n_per_proc = math.ceil(n / accelerator.num_processes)
    feats_all = []
    for i in trange(0, n_per_proc, batch_size, disable=not accelerator.is_main_process):
        cur_batch_size = min(n - i, batch_size)
        samples = sample_fn(cur_batch_size)[:cur_batch_size]
        feats_all.append(accelerator.gather(extractor_fn(samples)))
    return torch.cat(feats_all)[:n]
