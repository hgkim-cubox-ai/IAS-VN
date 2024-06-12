import torch
import torch.nn as nn
from torchvision.transforms.functional import rgb_to_grayscale as rgb2gray

from .backbone import CNN_ResNet


# Forwarding only
class LBPKernel(nn.Module):
    def __init__(self, batch_size=1):
        super(LBPKernel, self).__init__()
        
        kernel_weight = torch.tensor(
            [
                [[[0,0,1], [0,-1,0], [0,0,0]]], # top-right
                [[[0,0,0], [0,-1,1], [0,0,0]]], # top
                [[[0,0,0], [0,-1,0], [0,0,1]]], # bottom-right
                [[[0,0,0], [0,-1,0], [0,1,0]]], # bottom
                [[[0,0,0], [0,-1,0], [1,0,0]]], # bottom-left
                [[[0,0,0], [1,-1,0], [0,0,0]]], # left
                [[[1,0,0], [0,-1,0], [0,0,0]]], # top-left
                [[[0,1,0], [0,-1,0], [0,0,0]]]  # top
            ],
            dtype=torch.float32
        )
        self.lbp_kernel = torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, 
                                          padding=1, bias=False)
        self.lbp_kernel.weight.data = kernel_weight
        for param in self.lbp_kernel.parameters():
            param.requires_grad = False
        self.kernel_weight = torch.tensor([[[[1]], [[2]], [[4]], [[8]], [[16]], [[32]], [[64]], [[128]]]])
        self.lbp_hist = torch.zeros([batch_size, 256], dtype=torch.float32)
    
    def make_histogram(self, lbp_img):
        """
        lbp_img: NxHxW
        lbp_hist: Nx256
        """
        for i in range(len(lbp_img)):   # each image in batch
            for j in range(256):
                self.lbp_hist[i, j] = len(torch.where(j==lbp_img[i].view(-1))[0])
    
    def forward(self, img):
        img = rgb2gray(img)
        device = img.device
        
        # Forward
        out = self.lbp_kernel(img)
        out = torch.where(out >= 0.0, 1.0, 0.0)
        out = out.to(device)
        
        out = out * self.kernel_weight.to(device)
        lbp_img = torch.sum(out, dim=1).type(torch.float32)
        _ = self.make_histogram(lbp_img)
        self.lbp_hist = self.lbp_hist.to(device)
        lbp_hist = self.lbp_hist[:len(img)]
        
        # Normalize & reshape
        mean = torch.mean(lbp_hist, dim=1, keepdim=True)
        std = torch.std(lbp_hist, dim=1, keepdim=True)
        lbp_hist = (lbp_hist - mean) / std
        mean = torch.mean(lbp_img, dim=(1,2), keepdim=True)
        std = torch.std(lbp_img, dim=(1,2), keepdim=True)
        lbp_img = (lbp_img - mean) / std
        lbp_img = torch.unsqueeze(lbp_img, dim=1)
        
        return lbp_hist.detach(), lbp_img.detach()


class LBPModel(nn.Module):
    def __init__(self, cfg):
        super(LBPModel, self).__init__()
        
        self.cfg = cfg
        self.backbone = CNN_ResNet(cfg['backbone'])
        self.make_regressor()
        self.lbp_layer = LBPKernel(cfg['Data']['batch_size'])
        self.fc_lbp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def make_regressor(self):
        fc_in = 2048+256 if self.cfg['backbone'] > 'resnet18' else 512+256
        
        regressor = []
        regressor.append(nn.Linear(fc_in, self.cfg['regressor'][0]))
        for i in range(len(self.cfg['regressor'])-1):
            regressor.append(nn.ReLU(inplace=True))
            regressor.append(nn.Linear(self.cfg['regressor'][i],self.cfg['regressor'][i+1]))
        regressor.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*regressor)
            
    def forward(self, img):
        feat = self.backbone(img)
        feat = feat.view(feat.size(0), -1)  # flatten
        
        lbp_hist, lbp_img = self.lbp_layer(img)
        lbp_hist = self.fc_lbp(lbp_hist)
        feat = torch.cat([feat, lbp_hist], dim=1)
        
        out = self.regressor(feat)
        
        return out