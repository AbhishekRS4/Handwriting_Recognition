import torch.nn as nn
from typing import List
from torch import Tensor
from torchvision.models.resnet import BasicBlock, model_urls, load_state_dict_from_url, conv1x1, conv3x3


class CustomResNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        block=BasicBlock,
        zero_init_residual=False,
        groups=1,
        num_classes=1000,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):

        super().__init__()

        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 1), dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 1), dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 1), dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resnet(layers: List[int], pretrained=True) -> CustomResNet:
    model = CustomResNet(layers)

    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls["resnet34"]))

    return model

def resnet34(*, pretrained=True) -> CustomResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """

    return _resnet([3, 4, 6, 3], pretrained=pretrained)


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
            boolean to indicate whether to use a pretrained resnet model or not
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
