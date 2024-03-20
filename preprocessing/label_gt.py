import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import shutil
from glob import glob

from utils import is_image_file


# 여권, 운전면허증, 주민등록증 폴더
def copy_images_in_directory(path, label, output_path):
    all_files = os.listdir(path)
    img_files = [i for i in all_files if is_image_file(i)]
    img_idx = len(os.listdir(output_path))//2
    
    for img_file in img_files:
        ext = os.path.splitext(img_file)[1]
        shutil.copy(os.path.join(path, img_file),
                    os.path.join(output_path, f'{img_idx:08d}{ext}'))
        annot = {'spoof_label': label}
        with open(os.path.join(output_path, f'{img_idx:08d}.json'), 'w') as f:
            json.dump(annot, f, ensure_ascii=False)
        img_idx += 1
        
    
    # for img_path in img_paths:
    #     label_path = os.path.splitext(img_path)[0] + '.json'


def shinhan_poc():
    input_path = 'C:/Users/heegyoon/Desktop/data/IAS/raw/shinhan_poc/14시/크롭 이미지'
    output_path = 'C:/Users/heegyoon/Desktop/data/IAS/raw/shinhan'
    os.mkdir(output_path)
    sub_dirs1 = os.listdir(input_path)   # 사본 또는 원본 폴더
    label = None
    
    for sub_dir in sub_dirs1:
        sub_path1 = os.path.join(input_path, sub_dir)
        
        sub_dirs2 = os.listdir(sub_path1)
        for sub_dir2 in sub_dirs2:
            sub_path2 = os.path.join(sub_path1, sub_dir2)
            
            sub_dirs3 = os.listdir(sub_path2)
            for sub_dir3 in sub_dirs3:
                sub_path3 = os.path.join(sub_path2, sub_dir3)
                
                if sub_dir.endswith('사본'):
                    sub_dirs4 = os.listdir(sub_path3)
                    for sub_dir4 in sub_dirs4:
                        sub_path4 = os.path.join(sub_path3, sub_dir4)
                        label = 0
                        copy_images_in_directory(sub_path4, label, output_path)
                else:   # 진본
                    label = 1
                    copy_images_in_directory(sub_path3, label, output_path)
            
    
    
    
    print('')
    pass


def main():
    
    pass


if __name__ == '__main__':
    shinhan_poc()
    print('Done')