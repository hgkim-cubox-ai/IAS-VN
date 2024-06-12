import torch
import torch.nn as nn

from .backbone import CNN_ResNet

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from types_ import *


class IASModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(IASModel, self).__init__()
        
        self.cfg = cfg
        self.backbone = CNN_ResNet(cfg['backbone'])
        self.make_regressor()
    
    def make_regressor(self):
        fc_in = 2048 if self.cfg['backbone'] > 'resnet18' else 512
        
        regressor = []
        regressor.append(nn.Linear(fc_in, self.cfg['regressor'][0]))
        for i in range(len(self.cfg['regressor'])-1):
            regressor.append(nn.ReLU(inplace=True))
            regressor.append(nn.Linear(self.cfg['regressor'][i],self.cfg['regressor'][i+1]))
        regressor.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*regressor)
        
    
    def _forward_image(self, img):
        B = img.size(0)
        img = self.backbone(img)
        img = img.view(B, -1)
        img = self.regressor(img)
        return img
    
    def _forward_patch(self, patches):
        B, P, C, H, W = patches.size()
        patches = patches.view(B*P, C, H, W)
        if torch.isnan(patches).sum().item() > 0:
            print('')
        patches = self.backbone(patches)
        patches = patches.view(B*P, -1)
        patches = self.fc(patches)
        patches = patches.view(B, P, -1)
        patches = torch.mean(patches, dim=1)
        return patches

    def forward(self, x):
        if x.dim() == 4:
            return self._forward_image(x)
        elif x.dim() == 5:
            return self._forward_patch(x)
        else:
            raise ValueError

if __name__ == '__main__':
    net = IASModel({
        'backbone': 'resnet50',
        'regressor': [2048, 256, 16, 1]
    })
    # p = torch.randn([7, 9, 3, 128, 128])
    i = torch.randn([7, 3, 224, 224])
    out = net(i)
