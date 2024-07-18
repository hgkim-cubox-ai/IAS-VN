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


import cv2
import os
import numpy as np
import onnxruntime as ort

from typing import Optional

def align_idcard(img: np.ndarray, keypoints: np.ndarray, cls: float, dsize_factor: int = None) -> np.ndarray:
    if cls == 0:
        idcard_ratio = np.array((86, 54))
    elif cls == 1:
        idcard_ratio = np.array((125, 88))
    elif cls == 2:
        idcard_ratio = np.array((125, 88))
    else:
        idcard_ratio = np.array((640, 384))

    if dsize_factor is None:
        dsize_factor = round(np.sqrt(cv2.contourArea(np.expand_dims(keypoints, 1))) / idcard_ratio[0])

    dsize = idcard_ratio * dsize_factor  # idcard size unit: mm
    dst = np.array(((0, 0), (0, dsize[1]), dsize, (dsize[0], 0)), np.float32)

    M = cv2.getPerspectiveTransform(keypoints.astype(np.float32), dst)
    img = cv2.warpPerspective(img, M, dsize)

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

    masks = masks.astype(np.uint8)
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


class YOLOv8Seg:
    """YOLOv8 segmentation model."""

    def __init__(self, onnx_model):
        """
        Initialization.

        Args:
            onnx_model (str): Path to the ONNX model.
        """

        # Build Ort session
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        # Get model width and height(YOLOv8-seg only has one input)
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

        self.classes = [
            "kr-idcard",
            "kr-driver",
            "passport",
            "kr-alien-resident",
            "kr-permanent-resident",
            "kr-overseas-resident",
            "vn-cccd-nochip-front",
            "vn-cccd-nochip-back",
            "vn-cccd-chip-front",
            "vn-cccd-chip-back",
            "vn-cmnd-front",
            "vn-cmnd-back",
            "vn-driver-chip-front",
            "vn-driver-chip-back",
            "vn-passport"
        ]

        self.color_palette = [
            (255, 128, 0),
            (255, 153, 51),
            (255, 178, 102),
            (230, 230, 0),
            (0, 255, 255),
            (255, 153, 255),
            (153, 204, 255),
            (255, 102, 255),
            (255, 51, 255),
            (102, 178, 255),
            (51, 153, 255),
            (255, 153, 153),
            (255, 102, 102),
            (255, 51, 51),
            (153, 255, 153),
            (102, 255, 102),
            (51, 255, 51)]

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """

        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )
        return boxes, segments, masks

    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """

        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # Decode and return
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
            masks_boolean = np.greater(masks, 0.5)
            #masks = np.where(masks > 0.5, 255, 0).astype(np.uint8)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks_boolean)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []

    @staticmethod
    def masks2segments(masks):
        """
        It takes a list of masks(n,h,w) and returns a list of segments(n,xy) (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L750)

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return masks
        #return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def draw_and_visualize(self, im, bboxes, segments, vis=False, save=True, file_name=""):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """

        # Draw rectangles and polygons
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # draw contour and fill mask
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
            cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))

            #keypoints = get_keypoints(masks)

            # draw bbox rectangle
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.color_palette(int(cls_), bgr=True),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                im,
                f"{self.classes[cls_]}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_palette(int(cls_), bgr=True),
                2,
                cv2.LINE_AA,
            )

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # Show image
        if vis:
            cv2.imshow("demo", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image
        if save:
            if not file_name:
                file_name = "demo.jpg"
            cv2.imwrite(file_name, im)


def infer():
    detector = YOLOv8Seg('IDCard_Detection_20240403.onnx')
    data_dir = 'C:\\Users\\heegyoon\\Desktop\\data\\IAS\\vn\\raw\\Integration_Test'
    idcard_list = os.listdir(data_dir)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((144,224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,std=0.5)
    ])
    
    device = torch.device('cuda:0')
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
    ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for th in ths:
        results_dict = {
            'real': {
                'front': {'len': 0, 'correct': 0},
                'back': {'len': 0, 'correct': 0},
            },
            'laptop': {
                'front': {'len': 0, 'correct': 0},
                'back': {'len': 0, 'correct': 0},
            },
            'monitor': {
                'front': {'len': 0, 'correct': 0},
                'back': {'len': 0, 'correct': 0},
            },
            'paper': {
                'front': {'len': 0, 'correct': 0},
                'back': {'len': 0, 'correct': 0},
            },
            'smartphone': {
                'front': {'len': 0, 'correct': 0},
                'back': {'len': 0, 'correct': 0},
            },
        }
    
        for idcard in tqdm(idcard_list):
            idcard_dir = os.path.join(data_dir, idcard)
            img_list = os.listdir(idcard_dir)
            
            for img_file in img_list:
                fname = os.path.splitext(img_file)[0]
                try:
                    spoof_type, side = fname.split('_')
                    if not spoof_type in list(results_dict.keys()):
                        continue
                    if not side in ['front', 'back']:
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
                    img = align_idcard(img, keypoints, boxes[0][5])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results_dict[spoof_type][side]['len'] += 1
                    
                    img = np.transpose(img, [2,0,1])
                    img = torch.from_numpy(img)
                    
                    img = transform(img)
                    img = torch.unsqueeze(img, dim=0)
                    img = img.to(device)
                    
                    with torch.no_grad():
                        pred = model(img)
                    if pred.item() > th:
                        pred = 1.0
                    else:
                        pred = 0.0
                    
                    if spoof_type == 'real':
                        label = 1.0
                    else:
                        label = 0.0
                    
                    if pred == label:
                        results_dict[spoof_type][side]['correct'] += 1
        
        print(results_dict)
        log = []
        log.append(f'Threshold: {th}\n')
        for spoof_type, d1 in results_dict.items():
            for side, d2 in d1.items():
                l = d2['len']
                c = d2['correct']
                a = c / l * 100
                log.append(f'{spoof_type}_{side}: {a:.3f}\n')
        
        log_fname = 'infer_integration_test.txt'
        mode = 'a' if os.path.exists(log_fname) else 'w'
        with open(log_fname, mode, encoding='utf-8') as f:
            f.writelines(log)
            f.write('\n')
            


if __name__ == '__main__':
    # infer_pth()
    infer()
    print('Done')