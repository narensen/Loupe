import torch.nn as nn
import timm

class LoupeModule(nn.Module):
    def __init__(self, in_channels, reduction_factor=4):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_factor, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_factor, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.attention_net(x)

class SwinWithLoupe(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=200, pretrained=True):
        super().__init__()

        backbone = timm.create_model(model_name, pretrained=pretrained)

        self.patch_embed = backbone.patch_embed
        self.layers = backbone.layers
        self.norm = backbone.norm
        self.head = nn.Linear(backbone.head.in_features, num_classes)

        loupe_in_channels = self.layers[1].blocks[-1].norm1.normalized_shape[0]
        self.loupe = LoupeModule(in_channels=loupe_in_channels)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layers[0](x)
        x = self.layers[1](x)

        B, H, W, C = x.shape
        features_to_inspect = x.permute(0, 3, 1, 2)

        attention_map = self.loupe(features_to_inspect)
        refined_features = features_to_inspect * attention_map

        x = refined_features.permute(0, 2, 3, 1)
        x = self.layers[2](x)
        x = self.layers[3](x)

        x = x.flatten(1, 2)
        x = self.norm(x)
        x = x.mean(dim=1)

        logits = self.head(x)

        return logits, attention_map