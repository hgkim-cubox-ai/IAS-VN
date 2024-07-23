import torch.nn as nn
import torchvision.models as models


class CNN_ResNet(nn.Module):
    def __init__(self, backbone, weights=None):
        super(CNN_ResNet, self).__init__()
        
        if backbone == 'resnet18':
            model = models.resnet18(weights)
        elif backbone == 'resnet34':
            model = models.resnet34(weights)
        elif backbone == 'resnet50':
            model = models.resnet50(weights)
        elif backbone == 'resnet101':
            model = models.resnet101(weights)
        elif backbone == 'resnet152':
            model = models.resnet152(weights)
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


class ViT(nn.Module):
    def __init__(self, img_size, patch_size):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size