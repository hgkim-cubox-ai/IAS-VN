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
from models import LBPModel, ResNetModel
from utils import is_image_file
from idcard import align_idcard
from types_ import SPOOF_TYPE_DICT


def infer_pth():
    device = torch.device('cuda:0')
    
    # Data
    dataset = PersonData(
        {
            'data_path': 'C:/Users/heegyoon/Desktop/data/IAS/vn/dataset',
            'size': {'height': 144, 'width': 224},
            'datasets': {
                'test': ['TNGo_new3']
            }
        },
        is_train=False
    )
    loader = DataLoader(dataset, 1, False)
    
    # Model
    model = ResNetModel(
        {
            'backbone': 'resnet50',
            'regressor': [256, 16, 1],
        }
    )
    model = model.to(device)
    # Load weights
    tmp = torch.load('models/trained/baseline_res50_lr0.001_epoch51.pth')['state_dict']
    state_dict = OrderedDict()
    for n, v in tmp.items():
        state_dict[n[7:]] = v
    model.load_state_dict(state_dict)
    model.eval()
    
    # Thresholds
    ths = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    v2k_dict = {0: 'Real', 1: 'Laptop', 2: 'Monitor', 3: 'Paper', 4: 'SmartPhone'}
    
    for th in ths:
        log = []
        log.append(f'Threshold: {th}\n')
        results_dict = {}
        for key, value in SPOOF_TYPE_DICT.items():
            results_dict[key] = {
                'value': value,
                'len': 0,
                'correct': 0
            }
        
        for i, input_dict in enumerate(tqdm(loader, desc=f'Threshold: {th}')):
            img = input_dict['input'].to(device)
            label = input_dict['label'].item()
            spoof_type = v2k_dict[input_dict['spoof_type'].item()]
            results_dict[spoof_type]['len'] += 1
            
            # Infer
            with torch.no_grad():
                pred = model(img)
            if pred.item() > th:
                pred = 1.0
            else:
                pred = 0.0
            
            if pred == label:
                results_dict[spoof_type]['correct'] += 1
        
        # Calculate real/fake accuracy
        real_acc = (results_dict['Real']['correct'] / results_dict['Real']['len']) * 100
        fake_len = results_dict['Laptop']['len'] + results_dict['Monitor']['len'] + \
                   results_dict['Paper']['len'] + results_dict['SmartPhone']['len']
        fake_correct = results_dict['Laptop']['correct'] + results_dict['Monitor']['correct'] + \
                       results_dict['Paper']['correct'] + results_dict['SmartPhone']['correct']
        fake_acc = (fake_correct / fake_len) * 100
        log.append(f'real: {real_acc:.3f}\tfake: {fake_acc:.3f}\n')
        
        # Calculate spoof type accuracy
        for spoof_type, d in results_dict.items():
            l = d['len']; c = d['correct']
            acc = (c / l) * 100
            log.append(f'[{spoof_type}] acc: {acc:.3f}\n')
        
        print(results_dict)
        log_fname = 'infer_results.txt'
        mode = 'a' if os.path.exists(log_fname) else 'w'
        with open(log_fname, mode, encoding='utf-8') as f:
            f.writelines(log)
            f.write('\n')
            

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
    infer_pth()
    print('Done')