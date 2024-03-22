import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import cv2
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from typing import Optional

from backbone_code.idcard_segment import IDCardSegment
from utils import is_image_file


def get_image_paths(data_path):
    real_dirs = ('real_driver', 'real_id', 'real_passport')
    img_paths = []
    
    if data_path.endswith(real_dirs):
        sub_dirs = os.listdir(data_path)    
        for sub_dir in sub_dirs:
            img_paths += get_image_paths(os.path.join(data_path, sub_dir))
    else:
        img_paths = sorted(glob(os.path.join(data_path, '*.*')))
        img_paths = [i for i in img_paths if is_image_file(i)]
    
    return img_paths


def read_image(path):
    img = np.fromfile(path, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img


def sort_corner_order(quadrangle: np.ndarray) -> np.ndarray:
    assert quadrangle.shape == (4, 1, 2), f'Invalid quadrangle shape: {quadrangle.shape}'

    quadrangle = quadrangle.squeeze(1)
    moments = cv2.moments(quadrangle)
    mcx = round(moments['m10'] / moments['m00'])  # mass center x
    mcy = round(moments['m01'] / moments['m00'])  # mass center y
    keypoints = np.zeros((4, 2), np.int32)
    for point in quadrangle:
        if point[0] < mcx and point[1] < mcy:
            keypoints[0] = point
        elif point[0] < mcx and point[1] > mcy:
            keypoints[1] = point
        elif point[0] > mcx and point[1] > mcy:
            keypoints[2] = point
        elif point[0] > mcx and point[1] < mcy:
            keypoints[3] = point
    return keypoints


def get_keypoints(masks: np.ndarray, morph_ksize=21, contour_thres=0.02, poly_thres=0.03) -> Optional[np.ndarray]:
    # If multiple masks, select the mask with the largest object.
    if masks.shape[0] > 1:
        masks = masks[np.count_nonzero(masks.reshape(masks.shape[0], -1), axis=1).argmax()]

    # Post-process mask
    if len(masks.shape) == 3:
        masks = masks.squeeze(0)

    # Perform morphological transformation
    masks = cv2.morphologyEx(masks, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize)))
    # Find contours (+remove noise)
    contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = [contour for contour in contours
                if cv2.contourArea(contour) > (masks.shape[0] * masks.shape[1] * contour_thres)]
    # Approximate quadrangles (+remove noise)
    quadrangles = [cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * poly_thres, True) for contour in contours]
    quadrangles = [quad for quad in quadrangles if quad.shape == (4, 1, 2)]

    if len(quadrangles) == 1:
        keypoints = sort_corner_order(quadrangles[0])
        return keypoints
    else:
        return None


def align_idcard(img: np.ndarray, keypoints: np.ndarray, cls: list, dsize_factor: int = None) -> np.ndarray:
    if cls[0] == 0:
        idcard_ratio = np.array((86, 54))
    elif cls[0] == 1:
        idcard_ratio = np.array((125, 88))
    else:
        raise ValueError(f'Wrong cls: {cls}')

    if dsize_factor is None:
        dsize_factor = round(np.sqrt(cv2.contourArea(np.expand_dims(keypoints, 1))) / idcard_ratio[0])

    dsize = idcard_ratio * dsize_factor  # idcard size unit: mm
    dst = np.array(((0, 0), (0, dsize[1]), dsize, (dsize[0], 0)), np.float32)

    M = cv2.getPerspectiveTransform(keypoints.astype(np.float32), dst)
    img = cv2.warpPerspective(img, M, dsize)
    return img


def main():
    segmentor = IDCardSegment('preprocessing/detector.onnx', 0.8, 0.5, 'cuda')
    
    input_path = 'C:/Users/heegyoon/Desktop/data/IAS/vn/sample'
    data_dirs = os.listdir(input_path)
    # data_dirs.remove('CCCD 12')
    # data_dirs.remove('IBK_validset')
    # data_dirs.remove('shinhan_poc')
    # data_dirs = ['cubox_4k_2211']
    
    output_path = 'C:/Users/heegyoon/Desktop/data/IAS/vn/crop-resize'
    os.makedirs(output_path, exist_ok=True)
    
    blur_th = 200.0
    align = True
    check_blur = False

    for data_dir in data_dirs:
        img_paths = get_image_paths(os.path.join(input_path, data_dir))
        os.makedirs(os.path.join(output_path,data_dir), exist_ok=True)
        
        total_num = len(img_paths)
        img_idx = 0
        failed_imgs = []
        num_real = 0
        num_fake = 0
        annot_keys = set()
        kernel_size = 1 # default to 1 in cv2.laplacian()
        
        for img_path in tqdm(img_paths, f'{data_dir}'):
            label_path = os.path.splitext(img_path)[0] + '.json.plus_fd'
            try:
                with open(label_path, 'r') as f:
                    label = json.load(f)
            except:
                label = {'spoof_label': 1}
            
            label['before_processing'] = img_path
            
            annot_keys = annot_keys.union(set(label.keys()))
            # if label['spoof_type'] == '3.신분증+사진부착':
            #     failed_imgs.append(f'{img_path}, 신분증+사진\n')
            #     continue
            
            img = read_image(img_path)
                
            pred = segmentor.segment_one(img)
            if pred is not None:
                bbox, conf, cls, masks = pred
                keypoints = get_keypoints(masks)
                if keypoints is not None:
                    top = int(max(keypoints[0,1], keypoints[3,1]))
                    bottom = int(min(keypoints[1,1], keypoints[2,1]))
                    left = int(max(keypoints[0,0], keypoints[1,0]))
                    right = int(min(keypoints[2,0], keypoints[3,0]))
                    img_crop = img[top:bottom, left:right, :]
                    label['to_crop'] = [top, bottom, left, right]
                    
                    if check_blur:
                        # Blur check
                        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                        blur = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size).var()
                        if blur < blur_th:
                            failed_imgs.append(f'{img_path}, blurry\n')
                            continue
                        label['blur_var'] = blur
                    
                    if align:
                        aligned_img = align_idcard(img, keypoints, cls)
                        out = os.path.join(output_path, data_dir)
                        name = os.path.basename(os.path.splitext(img_path)[0])
                        cv2.imwrite(os.path.join(out ,os.path.basename(img_path)),
                                    aligned_img)
                        with open(os.path.join(out,name + '.json'), 'w') as f:
                            json.dump(label, f, ensure_ascii=False, indent=4)
                        
                    
                    if label['spoof_label'] == 1:
                        num_real += 1
                    else:
                        num_fake += 1
                    
                    # # Save the crop image & json.
                    # ext = os.path.splitext(img_path)[1]
                    # dst = os.path.join(output_path, data_dir, f'{img_idx:08d}{ext}')
                    # shutil.copy(img_path, dst)
                    # ###########################################################
                    # # debugging
                    # saved = cv2.imread(dst)
                    # if len(np.where(img!=saved)[0]):
                    #     print(img_path)
                    # ###########################################################
                    # with open(os.path.join(output_path, data_dir, f'{img_idx:08d}.json'), 'w') as f:
                    #     json.dump(label, f, ensure_ascii=False, indent=4)
                    img_idx += 1
                else:
                    failed_imgs.append(f'{img_path}, keypoints none\n')
            else:
                failed_imgs.append(f'{img_path}, pred none\n')

        with open(os.path.join(output_path, data_dir, 'log.txt'), 'w', encoding='utf-8') as f:
            f.write(f'Annotation keys: {list(annot_keys)}\n')
            f.write(f'total number of images: {total_num}\n')
            f.write(f'number of failures: {len(failed_imgs)}\n')
            f.write(f'number of real images: {num_real}\n')
            f.write(f'number of fake images: {num_fake}\n')
            f.writelines(failed_imgs)


if __name__ == '__main__':
    main()
    print('Done')

    '''
    conda env ias
    run in preprocessing folder
    '''