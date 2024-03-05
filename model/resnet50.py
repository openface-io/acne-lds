"""Contains resnet architectures."""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch.nn.functional as F


__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """Set 3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0)


class ResNet(nn.Module):
    """ResNet architecture class."""

    def __init__(self, block, layers, num_classes=1000, num_acne_cls=4):
        """Initialize layers, parameters etc."""
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
        self.num_acne_cls = num_acne_cls
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512 * block.expansion, self.num_acne_cls)
        self.counting = nn.Linear(512 * block.expansion, 65)

        self.counting.apply(weights_init)
        self.fc.apply(weights_init)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.bn1.apply(set_bn_fix)
        self.layer1.apply(set_bn_fix)
        self.layer2.apply(set_bn_fix)
        self.layer3.apply(set_bn_fix)
        self.layer4.apply(set_bn_fix)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Make a sequential container."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Calculate outputs."""
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

        cls = self.fc(x)
        cou = self.counting(x)

        cls = F.softmax(cls, dim=1) + 1e-4
        cou = F.softmax(cou, dim=1) + 1e-4

        cou2cls = torch.stack(
            (
                torch.sum(cou[:, :5], 1),
                torch.sum(cou[:, 5:20], 1),
                torch.sum(cou[:, 20:50], 1),
                torch.sum(cou[:, 50:], 1),
            ),
            1,
        )

        return cls, cou, cou2cls

    def train(self, mode=True):
        """Override train so that the training mode is set as we want."""
        nn.Module.train(self, mode)
        if mode:

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.eval()

            self.bn1.apply(set_bn_eval)
            self.layer1.apply(set_bn_eval)
            self.layer2.apply(set_bn_eval)
            self.layer3.apply(set_bn_eval)
            self.layer4.apply(set_bn_eval)


def resnet18(pretrained=False, **kwargs):
    """Construct a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Construct a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(num_acne_cls=4, **kwargs):
    """Construct a ResNet-50 model.

    Args:
        num_acne_cls: int, number of acne classes
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_acne_cls=num_acne_cls, **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Construct a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Construct a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model
