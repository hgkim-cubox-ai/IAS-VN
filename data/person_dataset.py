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


class PersonData(Dataset):
    def __init__(self, cfg, is_train: bool = True):
        self.cfg = cfg
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (cfg['size']['height'], cfg['size']['width']),
                transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        
        self.img_paths = []
        if is_train:
            dataset_list = cfg['datasets']['train']
        else:
            dataset_list = cfg['datasets']['test']
        for dataset in dataset_list:
            paths = glob(os.path.join(cfg['data_path'], dataset, '*/*.*'))
            self.img_paths += [i for i in paths if is_image_file(i)]
        self.json_paths = [os.path.splitext(i)[0]+'.json' for i in self.img_paths]            
        
    def __len__(self):
        return len(self.img_paths)
    
    def read_data(self, idx):
        img = np.fromfile(self.img_paths[idx], np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        with open(self.json_paths[idx], 'r', encoding='utf-8') as f:
            annots = json.load(f)
        return img, annots
        
    def preprocess_input(self, img, annots):
        img = align_idcard(img, annots['keypoints'])
        img = img[:, :, ::-1].copy()    # BGR to RGB
        img = self.transform(img)
        label = 1.0 if annots['spoof_label'] == 'Real' else 0.0
        return {'input': img, 'label': torch.tensor(label).float()}
        
    
    def __getitem__(self, idx):
        img, annots = self.read_data(idx)
        return self.preprocess_input(img, annots)


if __name__ == '__main__':
    dataset = PersonData(
        {
            'data_path': 'C:/Users/heegyoon/Desktop/data/IAS/vn/dataset',
            'patch_size': 256,
            'n_patches': 9,
            'size': {'height': 144, 'width': 224},
            'input': 'image',
            'color': 'rgb',
            'type': 'aligned',
            'datasets': {
                'train': ['TNGo_new', 'TNGo_new2'],
                'test': ['TNG_Employee']
            }
        }
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, 1, True)
    
    for i, (input_dict) in enumerate(loader):
        print(input_dict['input'].size(), input_dict['label'])
