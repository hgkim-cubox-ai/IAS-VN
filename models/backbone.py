import torch.nn as nn
import torchvision.models as models


class CNN_ResNet(nn.Module):
    def __init__(self, backbone):
        super(CNN_ResNet, self).__init__()
        
        if backbone == 'resnet18':
            model = models.resnet18()
        elif backbone == 'resnet50':
            model = models.resnet50()
        elif backbone == 'resnet101':
            model = models.resnet101()
        elif backbone == 'resnet152':
            model = models.resnet152()
        else:
            raise ValueError('invalid model name')
        
        module_list = list(model.children())
        self.layer = nn.Sequential(*module_list[:-1])
        
    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    def __init__(self, args):
        super(DenseNet, self).__init__()
        pass
    
    def forward(self, x):
        pass