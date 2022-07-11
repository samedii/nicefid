from resize_right import resize
from torch import nn
from torchvision import models, transforms
from torchvision.models import feature_extraction
import warnings


class InceptionV3WFeatureExtractor(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        model = (
            models.inception_v3(pretrained=True).to(device).eval().requires_grad_(False)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.extractor = feature_extraction.create_feature_extractor(
                model, {"flatten": "out"}
            )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.size = (299, 299)

    def forward(self, x):
        if x.shape[2:4] != self.size:
            x = resize(x, out_shape=self.size, pad_mode="reflect")
        x = self.normalize(x)
        return self.extractor(x)["out"]
