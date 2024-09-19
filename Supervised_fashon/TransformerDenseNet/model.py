import torch
import torchvision.models as models
import torch.nn as nn
import transformers.utils
from base_model import ImageEncoder
from transformers import ViTForImageClassification

class ImageTransformer(nn.Module):
    def __init__(self, num_classes,base=None,freeze_base=False):
        super().__init__()
        self.transformer = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',num_labels=num_classes)

    def forward(self, image):
        # Pass through Vision Transformer
        outputs = self.transformer(pixel_values=image)

        return outputs.logits

