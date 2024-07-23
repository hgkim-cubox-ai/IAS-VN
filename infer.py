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
import torchvision.transforms as transforms

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
                'val': ['TNGo_new3']
            }
        },
        is_train=False
    )
    loader = DataLoader(dataset, 1, False)
    
    # Model
    model_path = 'models/trained/baseline_ce_res50_lr0.001_epoch98.pth'
    model = ResNetModel(
        {
            'backbone': 'resnet50',
            'regressor': [256, 16, 5],
        }
    )
    model = model.to(device)
    # Load weights
    tmp = torch.load(model_path)['state_dict']
    state_dict = OrderedDict()
    for n, v in tmp.items():
        state_dict[n[7:]] = v
    model.load_state_dict(state_dict)
    model.eval()
    
    # Thresholds
    ths = [0.3]
    
    v2k_dict = {0: 'Real', 1: 'Laptop', 2: 'Monitor', 3: 'Paper', 4: 'SmartPhone'}
    
    for th in ths:
        results = np.zeros([5,5], dtype=np.int32)
        
        for i, input_dict in enumerate(tqdm(loader, desc=f'Threshold: {th}')):
            img = input_dict['input'].to(device)
            rf_label = input_dict['label'].item()
            cls_label = input_dict['spoof_type'].item()
            
            # Infer
            with torch.no_grad():
                pred = model(img)
            cls_pred = (torch.max(pred.detach(), 1)[1]).item()
            
            # Fill results
            results[cls_label][cls_pred] += 1
        
        results = results.tolist()
        
        log = []
        log.append(f'{os.path.basename(model_path)}\n')
        log.append(f'Threshold: {th}\n')
        log.append('\tR\tL\tM\tP\tS\n')
        for i in range(len(results)):
            cls = v2k_dict[i][0]
            log.append(f'{cls}\t')
            for j in range(len(results[i])):
                log.append(f'{results[i][j]}\t')
            log.append('\n')

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


def infer_with_detector():
    from detect_idcard import YOLOv8Seg, align_idcard, get_keypoints
    
    detector = YOLOv8Seg('models\\trained\\IDCard_Detection_20240403.onnx')
    data_dir = 'C:\\Users\\heegyoon\\Desktop\\data\\IAS\\vn\\raw\\Integration_Test'
    idcard_list = sorted(os.listdir(data_dir))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((144,224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,std=0.5)
    ])
    
    device = torch.device('cuda:0')
    
    # Model
    model_path = 'models/trained/baseline_ce_res34_lr0.001_epoch100.pth'
    model = ResNetModel(
        {
            'backbone': 'resnet34',
            'regressor': [256, 16, 5],
        }
    )
    model = model.to(device)
    # Load weights
    tmp = torch.load(model_path)['state_dict']
    state_dict = OrderedDict()
    for n, v in tmp.items():
        state_dict[n[7:]] = v
    model.load_state_dict(state_dict)
    model.eval()
    
    cls_dict = {
        'real': 0,
        'laptop': 1,
        'monitor': 2,
        'paper': 3,
        'smartphone': 4
    }
    v2k_dict = {
        0: 'R_F', 1: 'R_B',
        2: 'L_F', 3: 'L_B',
        4: 'M_F', 5: 'M_B',
        6: 'P_F', 7: 'P_B',
        8: 'S_F', 9: 'S_B'
    }
    
    # Thresholds
    ths = [0.3]
    
    sig = torch.nn.Sigmoid()
    ps = []
    
    for th in ths:
        results = np.zeros([10,5], dtype=np.int32)
    
        for idcard in tqdm(idcard_list):
            idcard_dir = os.path.join(data_dir, idcard)
            img_list = sorted(os.listdir(idcard_dir))
            
            for img_file in img_list:
                fname = os.path.splitext(img_file)[0]
                try:
                    cls_label, side = fname.split('_')
                    if not cls_label in cls_dict:
                        print(f'{idcard}/{fname}')
                        continue
                    if not side in ['front', 'back']:
                        print(f'{idcard}/{fname}')
                        continue
                except:
                    print(f'{idcard}/{fname}')
                    continue
                
                img_path = os.path.join(idcard_dir, img_file)
                img = cv2.imread(img_path)
                boxes, segments, masks = detector(img, conf_threshold=0.25, iou_threshold=0.45)
                            
                if len(boxes) <= 0:
                    continue
                
                keypoints = get_keypoints(masks)
                if keypoints is not None:
                    np_img = align_idcard(img, keypoints, boxes[0][5])
                    img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                    cls_label = cls_dict[cls_label] * 2
                    if side == 'back':
                        cls_label += 1
                    
                    img = np.transpose(img, [2,0,1])
                    img = torch.from_numpy(img)
                    
                    img = transform(img)
                    img = torch.unsqueeze(img, dim=0)
                    img = img.to(device)
                    
                    # Infer
                    with torch.no_grad():
                        pred = model(img)
                    cls_pred = (torch.max(pred.detach(), 1)[1]).item()
                    
                    # if cls_label < 2:
                    if cls_pred == 0:
                        p = sig(pred.view(-1)) / torch.sum(sig(pred))
                        p_real = p[0].item()
                        p_fake = torch.sum(p[1:]).item()
                        # ps.append([p_real, p_fake])
                        ps.append(p.cpu().tolist())

                    # Fill results
                    results[cls_label][cls_pred] += 1
        
        ps = np.array(ps)
        results = results.tolist()
        
        log = []
        log.append(f'{os.path.basename(model_path)}\n')
        log.append(f'Threshold: {th}\n')
        log.append('\tR\tL\tM\tP\tS\tTotal\tAcc\n')
        for i in range(len(results)):
            cls = v2k_dict[i]
            log.append(f'{cls}\t')
            for j in range(len(results[i])):
                log.append(f'{results[i][j]}\t')
            t = sum(results[i])
            log.append(f'{t}\t')
            if i < 2:
                a = results[i][0] / t * 100
            else:
                a = sum(results[i][1:]) / t * 100
            log.append(f'{a:.1f}%\n')

        log_fname = 'infer_integration_results.txt'
        mode = 'a' if os.path.exists(log_fname) else 'w'
        with open(log_fname, mode, encoding='utf-8') as f:
            f.writelines(log)
            f.write('\n')
                    
        #             with torch.no_grad():
        #                 pred = model(img)
        #             if pred.item() > th:
        #                 pred = 1.0
        #             else:
        #                 pred = 0.0
                    
        #             if spoof_type == 'real':
        #                 label = 1.0
        #             else:
        #                 label = 0.0
                    
        #             if pred == label:
        #                 results_dict[spoof_type][side]['correct'] += 1
        
        # print(results_dict)
        # log = []
        # log.append(f'Threshold: {th}\n')
        # for spoof_type, d1 in results_dict.items():
        #     for side, d2 in d1.items():
        #         l = d2['len']
        #         c = d2['correct']
        #         a = c / l * 100
        #         log.append(f'{spoof_type}_{side}: {a:.3f}\n')
        
        # log_fname = 'infer_integration_results.txt'
        # mode = 'a' if os.path.exists(log_fname) else 'w'
        # with open(log_fname, mode, encoding='utf-8') as f:
        #     f.writelines(log)
        #     f.write('\n')
            


if __name__ == '__main__':
    # infer_pth()
    infer_with_detector()
    print('Done')