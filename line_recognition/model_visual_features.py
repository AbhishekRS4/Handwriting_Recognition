import torch.nn as nn
from torchvision.models import resnet34


class ResNetFeatureExtractor(nn.Module):
    """
    Defines Base ResNet-34 feature extractor
    """
    def __init__(self, pretrained=True):
        """
        ---------
        Arguments
        ---------
        pretrained : bool (default=True)
            boolean to control whether to use a pretrained resnet model or not
        """
        super().__init__()
        self.output_channels = 512
        self.resnet34 = resnet34(pretrained=pretrained)

    def forward(self, x):
        block1 = self.resnet34.conv1(x)
        block1 = self.resnet34.bn1(block1)
        block1 = self.resnet34.relu(block1)   # [64, H/2, W/2]

        block2 = self.resnet34.maxpool(block1)
        block2 = self.resnet34.layer1(block2)  # [64, H/4, W/4]
        block3 = self.resnet34.layer2(block2)  # [128, H/8, W/8]
        block4 = self.resnet34.layer3(block3)  # [256, H/16, W/16]
        resnet_features = self.resnet34.layer4(block4)  # [512, H/32, W/32]

        # [B, 512, H/32, W/32]
        return resnet_features
