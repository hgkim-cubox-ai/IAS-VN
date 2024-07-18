import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import cv2
import json
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import is_image_file
from idcard import align_idcard
from types_ import SPOOF_TYPE_DICT


class PersonData(Dataset):
    def __init__(self, cfg, is_train: bool = True):
        self.cfg = cfg 
        self.img_paths = []
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(
                    (cfg['size']['height'], cfg['size']['width']),
                    transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize(mean=0.5, std=0.5)
            ])
            dataset_list = cfg['datasets']['train']
        else:
            dataset_list = cfg['datasets']['val']
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(
                    (cfg['size']['height'], cfg['size']['width']),
                    transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize(mean=0.5, std=0.5)
            ])
        for dataset in dataset_list:
            paths = glob(os.path.join(cfg['data_path'], dataset, '*/*.*'))
            self.img_paths += [i for i in paths if is_image_file(i)]
        self.json_paths = [os.path.splitext(i)[0]+'.json' for i in self.img_paths]            
        
    def __len__(self):
        return len(self.img_paths)
    
    def read_data(self, img_path, json_path):
        img = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        with open(json_path, 'r', encoding='utf-8') as f:
            annots = json.load(f)
        return img, annots
        
    def preprocess_input(self, img, annots):
        data_dict = {}
        img = img[:, :, ::-1].copy()    # BGR to RGB
        img = align_idcard(img, annots['keypoints'])
        img = self.transform(img)
        if annots['spoof_label'] == 'Real':
            label = 1.0
            data_dict['spoof_type'] = 0
        else:
            label = 0.0
            data_dict['spoof_type'] = SPOOF_TYPE_DICT[annots['spoof_type']]
        data_dict['input'] = img
        data_dict['label'] = torch.tensor(label).float()
        return data_dict
        
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        json_path = self.json_paths[idx]
        
        img, annots = self.read_data(img_path, json_path)
        data_dict = self.preprocess_input(img, annots)
        
        data_dict['img_path'] = img_path
        data_dict['json_path'] = json_path
        
        return data_dict


if __name__ == '__main__':
    dataset = PersonData(
        {
            'data_path': 'C:/Users/heegyoon/Desktop/data/IAS/vn/dataset',
            'size': {'height': 144, 'width': 224},
            'datasets': {
                'train': [
                    'TNGo_new',
                    'TNGo_new2',
                    # 'TNGo_new3',
                    # 'TNG_Employee'
                ],
                'test': ['TNG_Employee']
            }
        },
        False
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, 1, True)
    
    for i, (input_dict) in enumerate(loader):
        # print(input_dict['input'].size(), input_dict['label'])
        a = 1

'''
TNG_Employee:   74명,  6746 images
TNGo_new:       363명, 3630 images
TNGo_new2:      431명, 4310 images
TNGo_new3:      320명, 3200 images
'''