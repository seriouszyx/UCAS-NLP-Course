from torch import nn
from torchvision import models


def my_resnet_50(pretrained=False):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.add_module('softmax', nn.Softmax(dim=-1))
    for name, m in model.named_parameters():
        if name.split('.')[0] != 'fc':
            m.requires_grad_(False)
    return model
