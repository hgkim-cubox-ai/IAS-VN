import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
from glob import glob

from utils import is_image_file


def TNGo_new():
    root_dir = 'C:\\Users\\heegyoon\\Desktop\\data\\IAS\\vn\\raw\\TNGo_new'
    img_dir_list = ['Real', 'Fake_Laptop', 'Fake_Monitor', 'Fake_Paper', 'Fake_SmartPhone']
    json_path_list = [os.path.join(root_dir, 'label', i+'.json') for i in img_dir_list]
    spoof_types = ['Laptop', 'Monitor', 'Paper', 'SmartPhone']
    
    for json_path in json_path_list:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Each image
        for d in data:
            annots = d['annotations'][0]['result']
            assert len(annots) <= 7
            
            log = []
            
            # Fetch image path
            tmp = d['data']['image']
            tmp = f'{os.path.basename(os.path.dirname(tmp))}\\{os.path.basename(tmp)}'
            
            # File exists & check image file
            img_path = os.path.join(root_dir, tmp)
            if not os.path.exists(img_path):
                log.append('File not exists.\n')
            if not is_image_file(img_path):
                log.append('Not image file (wrong extension).\n')
            
            # Make personal label
            label = {}
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
                # Idcard type
                if annot['from_name'] == 'label':
                    label['idcard_type'] = annot['value']['polygonlabels'][0]
                    label['segmentation'] = annot['value']['points']
                    label['img_width']  = annot['original_width']
                    label['img_height'] = annot['original_height']
                # Device
                if annot['from_name'] == 'device_name':
                    label['device_name'] = annot['value']['text'][0]
                # Quality type
                if annot['from_name'] == 'quality_type':
                    label['quality_type'] = annot['value']['choices'][0]
            
            # Check label
            if not 'spoof_label' in label:
                log.append('Spoof_label is None.\n')
            # if not 'quality_type' in label:
            #     log.append(f'Quality_type is None.\n')
            if not 'spoof_type' in label:
                log.append('Spoof_type is None.\n')
            if not 'person_name' in label:
                log.append('Person_name is None.\n')
            if ('person_name' in label) and (len(label['person_name']) < 5):
                tmp1 = label['person_name']
                log.append(f'Wrong person_name. Person_name: {tmp1}\n')
            if not 'idcard_type' in label:
                log.append('Idcard_type is None.\n')
            if not 'segmentation' in label:
                log.append('Segmentation is None.\n')
            if label['spoof_label'] == 'Real':
                if label['spoof_type'] != 'None':
                    log.append('Real image has spoof_type.\n')
                if ('device_name' in label) and (label['device_name'] != 'none'):
                    log.append('Real iamge has device_name.\n')
            if label['spoof_label'] == 'Fake':
                if ('spoof_type' in label) and (not label['spoof_type'] in spoof_types):
                    tmp1 = label['spoof_label']; tmp2 = label['spoof_type']
                    log.append(f'Wrong spoof_type. Spoof_label: {tmp1}. Spoof_type: {tmp2}\n')

            if len(log) > 0:
                id = d['id']
                log.insert(0, f'ID: {id}, {tmp}\n')
                log.append('------------------------------------\n')
            
            # Save log
            log_path = os.path.join(root_dir, 'log_check_json.txt')
            mode = 'a' if os.path.exists(log_path) else 'w'
            with open(log_path, mode, encoding='utf-8') as f:
                f.writelines(log)
            

if __name__ == '__main__':
    TNGo_new()
    print('Done')

'''
data[0]['annotations'][0]['result']는 최대 6개
d = data[i]
annots = d['annotations'][0]['result']
annots의 'from_name': 'label'           # idcard segmentation (polygon)
                      'spoof_label'     # Real, Fake
                      'quality_type'    # good, bad, ...
                      'spoof_type'      # None, Paper, SmartPhone, ...
                      'device_name'     # iphone12, galaxy20, ...
                      'person_name'     # owner of idcard

d['data']['image']: path to image file
'''