import torch
import torch.nn as nn
from torchvision.transforms.functional import rgb_to_grayscale as rgb2gray

from .backbone import CNN_ResNet


class LBPKernel(nn.Module):
    def __init__(self):
        super(LBPKernel, self).__init__()
        
        self.lbp_kernel = torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, 
                                          padding=1, bias=False)
        self.lbp_kernel.weight.data = torch.tensor(
            [
                [[[0,0,1], [0,-1,0], [0,0,0]]], # top-right
                [[[0,0,0], [0,-1,1], [0,0,0]]], # right
                [[[0,0,0], [0,-1,0], [0,0,1]]], # bottom-right
                [[[0,0,0], [0,-1,0], [0,1,0]]], # bottom
                [[[0,0,0], [0,-1,0], [1,0,0]]], # bottom-left
                [[[0,0,0], [1,-1,0], [0,0,0]]], # left
                [[[1,0,0], [0,-1,0], [0,0,0]]], # top-left
                [[[0,1,0], [0,-1,0], [0,0,0]]]  # top
            ],
            dtype=torch.float32
        )
        self.kernel_weight = nn.parameter.Parameter(
            torch.tensor([[[[1]], [[2]], [[4]], [[8]], [[16]], [[32]], [[64]], [[128]]]], dtype=torch.float32)
        )

    def forward(self, img):
        img = rgb2gray(img)
        
        # LBP image
        out = self.lbp_kernel(img)
        out = torch.where(out >= 0.0, 1.0, 0.0)
        out = out * self.kernel_weight
        lbp_img = torch.sum(out, dim=1).type(torch.float32)
        
        lbp_hist = torch.zeros([1, 256], dtype=torch.float32)
        
        # LBP histogram
        for i in range(256):
            lbp_hist[0, i] = (torch.where(i==lbp_img)[0]).shape[0]
        # lbp_hist = self.lbp_hist
        
        # Normalize & reshape
        mean = torch.mean(lbp_hist, dim=1, keepdim=True)
        std = torch.std(lbp_hist, dim=1, keepdim=True)
        lbp_hist = (lbp_hist - mean) / std
        
        return lbp_hist


class LBPModel(nn.Module):
    def __init__(self):
        super(LBPModel, self).__init__()
        
        # Backbone
        self.backbone = CNN_ResNet('resnet50')
        # LBP kernel
        self.lbp_layer = LBPKernel()
        self.fc_lbp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # regressor
        fc = [2048, 256, 16, 1]
        self.make_regressor(fc)

    def make_regressor(self, fc):        
        regressor = []
        regressor.append(nn.Linear(2304, fc[0]))
        for i in range(len(fc)-1):
            regressor.append(nn.ReLU(inplace=True))
            regressor.append(nn.Linear(fc[i],fc[i+1]))
        regressor.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*regressor)
            
    def forward(self, img):
        feat = self.backbone(img)
        feat = feat.view(feat.size(0), -1)  # flatten
        
        lbp_hist = self.lbp_layer(img)
        lbp_hist = self.fc_lbp(lbp_hist)
        
        feat = torch.cat([feat, lbp_hist], dim=1)
        out = self.regressor(feat)
        
        return out