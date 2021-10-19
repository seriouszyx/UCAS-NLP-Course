from torch import nn
from torchvision import models

from MyCNNNet import MyCNNNet
from MyDNNNet import MyDNNNet
from MyResNet34 import MyResNet34


def my_resnet_50(pretrained=False):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.add_module('softmax', nn.Softmax(dim=-1))
    for name, m in model.named_parameters():
        if name.split('.')[0] != 'fc':
            m.requires_grad_(False)
    return model


def my_cnnnet():
    model = MyCNNNet()
    model.add_module('softmax', nn.Softmax(dim=-1))
    return model


def my_resnet_34():
    model = MyResNet34()
    model.add_module('softmax', nn.Softmax(dim=-1))
    return model

def my_dnnnet():
    model = MyDNNNet()
    model.add_module('softmax', nn.Softmax(dim=-1))
    return model