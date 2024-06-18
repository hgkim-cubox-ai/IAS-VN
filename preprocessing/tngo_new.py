import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import shutil
import cv2
import numpy as np

from utils import is_image_file
from idcard import get_keypoints


SPOOF_TYPE_DICT = {
    'Real': {'front': 0, 'back': 1},
    'Fake': {
        'Laptop': {'front': 2, 'back': 3},
        'Monitor': {'front': 4, 'back': 5},
        'Paper': {'front': 6, 'back': 7},
        'SmartPhone': {'front': 8, 'back': 9},
    }
}
ROOT_DIR = 'C:\\Users\\heegyoon\\Desktop\\data\\IAS\\vn\\raw\\TNGo_new'


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
        'from_path': None,
        'keypoints': None,
        'segmentation': None
    }
    for annot in annots:
        # Name
        if annot['from_name'] == 'person_name':
            label['person_name'] = annot['value']['text'][0]
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
    
    # Fill None
    for k, v in label.items():
        if isinstance(v, str) and v.lower() == 'none':
            label[k] = None
    
    return label


def check_label(data):
    label = json_to_personal_label(data)
    # No name
    if label['person_name'] == None:
        return None
    # No idcard type
    if label['idcard_type'] == None:
        return None
    # Segmentation
    if label['segmentation'] == None:
        return None
    # Real image
    if label['spoof_label'] == 'Real':
        if (label['spoof_type'] != None) or (label['device_name'] != None):
            return None
    # Fake image
    else:
        if label['spoof_type'] == None:
            return None
    
    # File does not exits
    tmp = data['data']['image']
    dirname = os.path.basename(os.path.dirname(tmp))
    filename = os.path.basename(tmp)
    img_path = os.path.join(ROOT_DIR, dirname, filename)
    if not os.path.exists(img_path):
        return None
    # Not an image file
    if not is_image_file(img_path):
        return None
    
    label['from_path'] = dirname + '/' + filename

    return label


def make_filename(label):
    img_idx = None
    idcard_type = label['idcard_type'].split('-')[-1]   # front or back
    if label['spoof_label'] == 'Real':
        img_idx = SPOOF_TYPE_DICT['Real'][idcard_type]
        filename = f'{img_idx}_real_' + label['idcard_type'].split('-')[-1]
    else:   # Fake
        img_idx = SPOOF_TYPE_DICT['Fake'][label['spoof_type']][idcard_type]
        filename = f'{img_idx}_fake_' + label['spoof_type'] + '_' + label['idcard_type'].split('-')[-1]
    assert img_idx != None
    
    return filename


def main():
    img_dir_list = ['Real', 'Fake_Laptop', 'Fake_Monitor', 'Fake_Paper', 'Fake_SmartPhone']
    json_path_list = [os.path.join(ROOT_DIR, 'label', i+'.json') for i in img_dir_list]
    src_dir = 'C:\\Users\\heegyoon\\Desktop\\data\\IAS\\vn\\tmp\\TNGo_new'
    os.makedirs(src_dir, exist_ok=True)
    
    for json_path in json_path_list:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Each image
        for d in data:
            label = check_label(d)
            if label == None:
                continue
            
            person_name = label['person_name']
            if person_name.startswith(' '):
                person_name = person_name[1:]
            if person_name.endswith(' '):
                person_name = person_name[:-1]
            os.makedirs(os.path.join(src_dir, person_name), exist_ok=True)
            
            src_dirname, src_filename = os.path.split(d['data']['image'])
            src_dirname = os.path.basename(src_dirname)
            src = os.path.join(ROOT_DIR, src_dirname, src_filename)
            if os.path.isfile(src) and is_image_file(src):
                dst_filename = make_filename(label)
                dst = os.path.join(src_dir, person_name, dst_filename + os.path.splitext(src)[1])
                json_path = os.path.join(src_dir, person_name, dst_filename + '.json')
                
                # Segmentation to keypoints
                W = label['image_width']; H = label['image_height']
                sgmnt = np.array(label['segmentation'])
                sgmnt[:, 0] = sgmnt[:, 0] * W / 100
                sgmnt[:, 1] = sgmnt[:, 1] * H / 100
                sgmnt = sgmnt.astype(np.int32)
                mask = np.zeros([H,W], dtype=np.uint8)
                cv2.fillPoly(mask, [sgmnt], 255)
                keypoints = get_keypoints(mask)
                if keypoints is None:
                    continue
                else:
                    label['keypoints'] = keypoints.tolist()
                
                # Move image & save json
                shutil.move(src, dst)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(label, f, ensure_ascii=False, indent=4)
    
    dst_dir = 'C:\\Users\\heegyoon\\Desktop\\data\\IAS\\vn\\dataset\\TNGo_new'
    person_list = os.listdir(src_dir)
    for person in person_list:
        src_person_dir = os.path.join(src_dir, person)
        
        files = os.listdir(src_person_dir)
        if len(files) == 20:
            dst_person_dir = os.path.join(dst_dir, person)
            shutil.move(src_person_dir, dst_person_dir)
         

if __name__ == '__main__':
    main()
    print('Done')