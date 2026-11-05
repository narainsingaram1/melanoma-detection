import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SkinLesionTransfer(nn.Module):
    """
    Transfer learning model using EfficientNet-B0 for Skin Lesion Classification.
    """
    def __init__(self, num_classes=7):
        super(SkinLesionTransfer, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Freeze all layers initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze the last two blocks (block 7 and 8 typically for EfficientNet, or features[-2:])
        # In efficientnet_b0, features contains 9 main modules (0 to 8). 
        # We unfreeze blocks 7 and 8.
        for param in self.backbone.features[7].parameters():
            param.requires_grad = True
        for param in self.backbone.features[8].parameters():
            param.requires_grad = True
            
        # Replace the classifier head
        # EfficientNet-B0 has a classifier: Dropout(p=0.2) -> Linear(in_features=1280, out_features=1000)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_all(self):
        """
        Unfreeze all layers for full fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_last_conv_layer(self):
        """
        Returns the name of the last convolutional layer for Grad-CAM.
        For EfficientNet, it's typically the final conv in features.
        """
        # The last conv layer is generally features.8 (Conv2dNormActivation)
        return "backbone.features.8"

if __name__ == "__main__":
    model = SkinLesionTransfer()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print("Transfer Model output shape:", out.shape)
