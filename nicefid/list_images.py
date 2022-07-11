from pathlib import Path

EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp", "npy"}


def list_images(path):
    return [
        image_path
        for image_path in Path(path).glob(f"**/*.*")
        if image_path.suffix[1:].lower() in EXTENSIONS
    ]
