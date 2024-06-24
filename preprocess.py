import os
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from urllib.parse import unquote

from utils import is_image_file
from idcard import get_keypoints


def convert_path(label, data_dir):
    drive_path = label['drive_path']
    dataset_name = os.path.basename(data_dir)
    
    if dataset_name == 'TNG_Employee':
        if label['spoof_type'] == 'Paper':
            dirname = os.path.basename(os.path.dirname(drive_path))
        else:
            dirname1 = label['person_name']
            dirname2 = os.path.basename(os.path.dirname(drive_path))
            dirname = os.path.join(dirname1, dirname2)        
    elif dataset_name == 'TNGo_new':
        dirname = os.path.basename(os.path.dirname(drive_path))
    elif dataset_name >= 'TNGo_new2':   # new2, new3, new4
        if label['spoof_label'] == 'Real':
            dirname1 = 'Real'
            dirname2 = unquote(os.path.basename(os.path.dirname(drive_path)))
            dirname = os.path.join(dirname1, dirname2)
        else:
            dirname = os.path.basename(os.path.dirname(drive_path))
    else:
        raise ValueError('Wrong dataset name.')
    filename = os.path.basename(drive_path)
    img_path = os.path.join(data_dir, dirname, filename)
    
    # File does not exits or wrong extenstion
    if not (os.path.exists(img_path) and is_image_file(img_path)):
        img_path = None
    
    return img_path


def json_to_personal_label(data):
    annots = data['annotations'][0]['result']
    assert len(annots) <= 7
    
    label = {
        'person_name': None,
        'spoof_label': None,
        'spoof_type': None,
        'idcard_type': None,
        'device_name': None,
        'quality_type': None,
        'image_width': None,
        'image_height': None,
        'drive_path': None,
        'image_path': None,
        'keypoints': None,
        'segmentation': None
    }
    # Fill label
    for annot in annots:
        # Name
        if annot['from_name'] == 'person_name':
            person_name = annot['value']['text'][0]
            
            # Remove space
            if person_name.startswith(' '):
                person_name = person_name[1:]
            if person_name.endswith(' '):
                person_name = person_name[:-1]
            label['person_name'] = person_name
        # Spoof label
        if annot['from_name'] == 'spoof_label':
            label['spoof_label'] = annot['value']['choices'][0]
        # Spoof type
        if annot['from_name'] == 'spoof_type':
            label['spoof_type'] = annot['value']['choices'][0]
        # Idcard type & segmentation
        if annot['from_name'] == 'label':
            label['idcard_type'] = annot['value']['polygonlabels'][0]
            label['segmentation'] = annot['value']['points']
            label['image_width'] = annot['original_width']
            label['image_height'] = annot['original_height']
        # Device
        if annot['from_name'] == 'device_name':
            label['device_name'] = annot['value']['text'][0]
        # Quality type
        if annot['from_name'] == 'quality_type':
            label['quality_type'] = annot['value']['choices'][0]
    label['drive_path'] = data['data']['image']
    
    # Convert 'None' to None
    for k, v in label.items():
        if isinstance(v, str) and v.lower() == 'none':
            label[k] = None    
        
    return label


def check_label(label):
    # No name
    if label['person_name'] == None:
        return None
    # No idcard type
    if label['idcard_type'] == None:
        return None
    # Segmentation
    if label['segmentation'] == None:
        return None
    # Spoof label
    if label['spoof_label'] == None:
        return None
    # Real image
    if label['spoof_label'] == 'Real':
        if (label['spoof_type'] != None) or (label['device_name'] != None):
            return None
    # Fake image
    else:
        if label['spoof_type'] == None:
            return None

    return label


def add_annotations(label, data_dir):
    # Image path in local
    img_path = convert_path(label, data_dir)
    if img_path == None:
        return None
    label['image_path'] = img_path
    
    # Segmentation
    W = label['image_width']; H = label['image_height']
    sgmnt = np.array(label['segmentation'])
    sgmnt[:, 0] = sgmnt[:, 0] * W / 100
    sgmnt[:, 1] = sgmnt[:, 1] * H / 100
    sgmnt = sgmnt.astype(np.int32)
    mask = np.zeros([H,W], dtype=np.uint8)
    cv2.fillPoly(mask, [sgmnt], 255)
    keypoints = get_keypoints(mask)
    if keypoints is None:
        return None
    label['keypoints'] = keypoints.tolist()
    
    return label


def make_filname(label, dst_dir):
    spoof_label = label['spoof_label'].lower()
    side = label['idcard_type'].split('-')[-1]  # front or back
    
    if spoof_label == 'real':
        prefix = f'{spoof_label}_{side}'
    else:   # Fake
        spoof_type = label['spoof_type'].lower()
        prefix = f'{spoof_label}_{spoof_type}_{side}'
    
    img_idx = 0
    img_files = [i for i in os.listdir(dst_dir) if is_image_file(i)]
    for img_file in img_files:
        if prefix in img_file:
            img_idx += 1
    
    return f'{prefix}_{img_idx}'


def main():
    root_dir = 'C:\\Users\\heegyoon\\Desktop\\data\\IAS\\vn'
    raw_dir = os.path.join(root_dir, 'raw')
    tmp_dir = os.path.join(root_dir, 'tmp')
    dst_dir = os.path.join(root_dir, 'dataset')
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    
    dataset_dict = {
        # 'TNG_Employee': ['fake_paper.json', 'others.json'],
        # 'TNGo_new': ['Fake_Laptop.json', 'Fake_Monitor.json', 'Fake_Paper.json',
        #              'Fake_SmartPhone.json', 'Real.json'],
        # 'TNGo_new2': ['Fake_Laptop.json', 'Fake_Monitor.json', 'Fake_Paper.json',
        #               'Fake_SmartPhone.json', 'Real.json'],
        'TNGo_new3': ['Laptop.json', 'Monitor.json', 'Paper.json', 'Real.json',
                      'SmartPhone.json'],
        # 'TNGo_new4': []
    }
    
    for dataset_name, json_files in dataset_dict.items():
        cur_dir = os.path.join(raw_dir, dataset_name)
        for json_file in json_files:
            json_path = os.path.join(raw_dir, dataset_name, json_file)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Each data
            for d in tqdm(data, desc=f'{dataset_name}/{json_file}'):
                label = json_to_personal_label(d)
                label = check_label(label)
                if label == None:
                    continue
                label = add_annotations(label, cur_dir)
                if label == None:
                    continue
                
                person_name = label['person_name']
                dst_dir = os.path.join(tmp_dir, dataset_name, person_name)
                os.makedirs(dst_dir, exist_ok=True)
                
                filename = make_filname(label, dst_dir)
                src_path = label['image_path']
                dst_path = os.path.join(dst_dir, filename + os.path.splitext(src_path)[1])
                json_path = os.path.join(dst_dir, filename + '.json')
                
                # Move image & save json
                shutil.copy(src_path, dst_path)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(label, f, ensure_ascii=False, indent=4)
            


if __name__ == '__main__':
    main()
    print('Done')