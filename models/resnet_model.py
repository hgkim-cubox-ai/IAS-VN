import torch
import torch.nn as nn

from .backbone import CNN_ResNet


class ResNetModel(nn.Module):
    def __init__(self, cfg):
        super(ResNetModel, self).__init__()
        
        self.cfg = cfg
        self.backbone = CNN_ResNet(cfg['backbone'])
        self.make_regressor()
    
    def make_regressor(self):
        fc_in = 2048 if int(self.cfg['backbone'][6:]) > 34 else 512
        
        regressor = []
        regressor.append(nn.Linear(fc_in, self.cfg['regressor'][0]))
        for i in range(len(self.cfg['regressor'])-1):
            regressor.append(nn.ReLU(inplace=True))
            regressor.append(nn.Linear(self.cfg['regressor'][i],self.cfg['regressor'][i+1]))
        regressor.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*regressor)
    
    def forward(self, img):
        feat = self.backbone(img)
        feat = feat.view(feat.size(0), -1)
        feat = self.regressor(feat)
        return feat