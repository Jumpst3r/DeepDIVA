"""
Model definition adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import logging
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['hqresnet18', 'hqresnet34', 'hqresnet50', 'hqresnet101',
           'hqresnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class _HQ_ResNet(nn.Module):

    def __init__(self, block, layers, output_channels=1000):
        # TODO: understand what that is and why it is always (for all layers)
        # TODO: set to 64.... at the moment set to 128 and it works
        self.inplanes = 128
        super(_HQ_ResNet, self).__init__()

        self.expected_input_size = (472, 472)

        # TODO: A new convolution and batch norm is defined
        self.conv00 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=0,
                               bias=False)
        self.bn00 = nn.BatchNorm2d(8)

        # TODO: A new convolution and batch norm is defined
        self.conv01 = nn.Conv2d(8, 16, kernel_size=7, stride=1, padding=0,
                                bias=False)
        self.bn01 = nn.BatchNorm2d(16)

        self.conv02 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=0,
                               bias=False)
        self.bn02 = nn.BatchNorm2d(32)

        self.conv03 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=0,
                               bias=False)
        self.bn03 = nn.BatchNorm2d(64)

        # try increasing the filters
        self.conv04 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn04 = nn.BatchNorm2d(64)

        # TODO: only the filter dimensions are changed now
        self.conv1 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024 * block.expansion, output_channels)
        print("***************************************************")
        print("NN init with : " + str(output_channels) + " output_channels")

        # todo: understand this part. Is it normalization and BatchNorm?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # TODO: this is the new layer that halves the image to
        # TODO: the size that was previously expected.
        x = self.conv00(x)
        x = self.bn00(x)
        x = self.relu(x)

        x = self.conv01(x)
        x = self.bn01(x)
        x = self.relu(x)

        x = self.conv02(x)
        x = self.bn02(x)
        x = self.relu(x)

        x = self.conv03(x)
        x = self.bn03(x)
        x = self.relu(x)

        x = self.conv04(x)
        x = self.bn04(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def hqresnet18(pretrained=False, **kwargs):
    """Constructs a _ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _HQ_ResNet(_BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def hqresnet34(pretrained=False, **kwargs):
    """Constructs a _ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _HQ_ResNet(_BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def hqresnet50(pretrained=False, **kwargs):
    """Constructs a _ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _HQ_ResNet(_Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def hqresnet101(pretrained=False, **kwargs):
    """Constructs a _ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _HQ_ResNet(_Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def hqresnet152(pretrained=False, **kwargs):
    """Constructs a _ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _HQ_ResNet(_Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model
