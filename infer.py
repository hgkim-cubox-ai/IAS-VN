import os
import onnx
import onnxruntime as ort
import cv2
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from data import PersonData
from models import LBPModel
from utils import is_image_file
from idcard import align_idcard


def infer_pth():
    device = torch.device('cuda:0')
    
    # Data
    dataset = PersonData(
        {
            'data_path': 'C:/Users/heegyoon/Desktop/data/IAS/vn/dataset',
            'size': {'height': 144, 'width': 224},
            'datasets': {
                'test': ['TNG_Employee']
            }
        },
        is_train=False
    )
    loader = DataLoader(dataset, 1, False)
    
    # Model
    model = LBPModel(
        {
            'backbone': 'resnet50',
            'regressor': [2048, 256, 16, 1],
            'Data': {'batch_size': 1}
        }
    )
    model = model.to(device)
    # Load weights
    tmp = torch.load('models/ias_model.pth')['state_dict']
    state_dict = OrderedDict()
    for n, v in tmp.items():
        state_dict[n[7:]] = v
    model.load_state_dict(state_dict)
    model.eval()
    
    # Thresholds
    ths = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
           0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    for th in ths:
        results_dict = {
            0: {'len': 0, 'correct': 0},    # fake
            1: {'len': 0, 'correct': 0}     # real
        }
        
        for i, input_dict in enumerate(tqdm(loader, desc=f'{th}')):
            img = input_dict['input'].to(device)
            label = input_dict['label']
            results_dict[int(label.item())]['len'] += 1
            
            with torch.no_grad():
                pred = model(img)
            pred = torch.where(pred > th, 1, 0).cpu()
            
            if pred.item() == label.item():
                results_dict[int(label.item())]['correct'] += 1
        
        # Calculate accuracy    
        real_acc = (results_dict[1]['correct'] / results_dict[1]['len']) * 100
        fake_acc = (results_dict[0]['correct'] / results_dict[0]['len']) * 100
        print(results_dict)
        
        # Save log
        log = []
        log.append(f'Threshold: {th}\n')
        log.append(f'real: {real_acc:.3f}\tfake: {fake_acc:.3f}\n\n')
        log_name = 'demo_results2.txt'
        mode = 'a' if os.path.exists(log_name) else 'w'
        with open(log_name, mode, encoding='utf-8') as f:
            f.writelines(log)


def infer_onnx():
    so = ort.SessionOptions()
    providers = ['CUDAExecutionProvider']
    session = ort.InferenceSession(
        'models/ias_infer.onnx',
        so,
        providers=providers
    )
    
    data_dir = 'C:/Users/heegyoon/Desktop/data/IAS/vn/dataset/TNG_Employee'
    paths = glob(os.path.join(data_dir, '*', '*.*'))
    img_paths = sorted([i for i in paths if is_image_file(i)])
    json_paths = sorted([os.path.splitext(i)[0]+'.json' for i in img_paths])
    
    results_dict = {
        0: {'len': 0, 'correct': 0},    # fake
        1: {'len': 0, 'correct': 0}     # real
    }
    th = 0.4
    
    for img_path, json_path in tqdm(zip(img_paths, json_paths)):
        assert os.path.splitext(img_path)[0] == os.path.splitext(json_path)[0]
        # Read data
        img = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        with open(json_path, 'r', encoding='utf-8') as f:
            annots = json.load(f)
        label = 1 if annots['spoof_label'] == 'Real' else 0
        results_dict[label]['len'] += 1
        
        # Preprocess data
        img = align_idcard(img, annots['keypoints'])
        img = img[:, :, ::-1].copy()    # BGR to RGB
        img = cv2.resize(img, (224, 144), interpolation=cv2.INTER_CUBIC)
        img = 2 * img.astype(np.float32) / 255 - 1
        img = np.transpose(img, [2,0,1])
        img = np.expand_dims(img, axis=0)
        
        outputs = session.run(['output'], {'input': img})
        tmp = outputs[0][0,0]
        if tmp > th:
            pred = 1
        else:
            pred = 0
        
        if pred == label:
            results_dict[label]['correct'] += 1
            
    # Calculate accuracy    
    real_acc = (results_dict[1]['correct'] / results_dict[1]['len']) * 100
    fake_acc = (results_dict[0]['correct'] / results_dict[0]['len']) * 100
    print(results_dict)
    
    # Save log
    log = []
    log.append(f'Threshold: {th}\n')
    log.append(f'real: {real_acc:.3f}\tfake: {fake_acc:.3f}\n\n')
    log_name = 'infer_onnx_results.txt'
    mode = 'a' if os.path.exists(log_name) else 'w'
    with open(log_name, mode, encoding='utf-8') as f:
        f.writelines(log)


if __name__ == '__main__':
    infer_onnx()
    print('Done')