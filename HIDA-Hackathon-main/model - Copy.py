from torch import nn
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2


def MaskRCNN(in_channels=5, num_classes=2, image_mean=None, image_std=None, **kwargs):
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225, 0.225, 0.225]

    model = maskrcnn_resnet50_fpn_v2(
        num_classes=num_classes,
        image_mean=image_mean,
        image_std=image_std
    )
    model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).requires_grad_(True)

    return model

class Ensemble(nn.Module):
    def __init__(self, model_a, model_b, num_classes=10):
        super(Ensemble, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.num_classes = num_classes
        # You can also add a classifier layer here if you want to learn a combination strategy

    def forward(self, x):
        # Get predictions from each model
        output_a = self.model_a(x)
        output_b = self.model_b(x)
        
        # Average the predictions
        output = (output_a + output_b) / 2
        return output

def Ensemble_f(ModelA,ModelB):
    model_a = ModelA
    model_b = ModelB

    # Initialize ensemble model
    ensemble_model = Ensemble(model_a, model_b)

    return ensemble_model