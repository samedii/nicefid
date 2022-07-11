# nicefid

Minimalistic FID and KID implementation. Reference checked against [cleanfid](https://github.com/GaParmar/clean-fid).
Code is a mix between [crowsonkb's implementation](https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/evaluation.py)
and [cleanfid](https://github.com/GaParmar/clean-fid).

> [On Aliased Resizing and Surprising Subtleties in GAN Evaluation](https://arxiv.org/abs/2104.11222)

## Install

```bash
poetry add nicefid
```

Or, for the old timers:

```bash
pip install nicefid
```

## API

```python
nicefid.Features.from_directory(path: Union[str, Path])
nicefid.Features.from_iterator(iterator: Iterator[torch.Tensor])  # NCHW
nicefid.Features.from_path(path: Union[str, Path])
features.save(path: Union[str, Path])

nicefid.compute_fid(features_a, features_b)
nicefid.compute_kid(features_a, features_b)
```

## Usage

Comparing directory with generated images.

```python
import nicefid

features_generated = nicefid.Features.from_iterator(...)
features_real = nicefid.Features.from_directory(...)

fid = nicefid.compute_fid(features_generated, features_real)
kid = nicefid.compute_kid(features_generated, features_real)
```
