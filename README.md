# nicefid

Minimalistic wrapper around [cleanfid](https://github.com/GaParmar/clean-fid) to make the api more user friendly. Calculates FID and KID from sets of images.

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
nicefid.Features.from_generator(generator: Callable[[], torch.Tensor])  # NCHW
nicefid.Features.from_name(name: str)

features.save(name: str)
nicefid.saved_features_exists(name: str) -> bool
nicefid.remove_saved_features(name: str)

nicefid.compute_fid(features_a, features_b)
nicefid.compute_kid(features_a, features_b)
```

## Usage

Comparing directory with generated images.

```python
import nicefid

features_generated = nicefid.Features.from_generator(...)
features_real = nicefid.Features.from_directory(...)

fid = nicefid.compute_fid(features_generated, features_real)
kid = nicefid.compute_kid(features_generated, features_real)
```

### Save features

```python
nicefid.Features.from_directory("test-dataset").save(name)
```
