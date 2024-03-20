import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import onnxruntime
import torch
import torchvision

from backbone_code.detect_segment_utils.general import clip_coords, resize_preserving_aspect_ratio, xywh2xyxy


class YOLO5Face:
    def __init__(self, model_path: str, conf_thres: float, iou_thres: float, device: str):
        """
        Args:
            model_path: Model file path.
            conf_thres: Confidence threshold.
            iou_thres: IoU threshold.
            device: Device to inference.
        """
        assert os.path.exists(model_path), f'model_path is not exists: {model_path}'
        assert 0 <= conf_thres <= 1, f'conf_thres must be between 0 and 1: {conf_thres}'
        assert 0 <= iou_thres <= 1, f'iou_thres must be between 0 and 1: {iou_thres}'
        assert device in ['cpu', 'cuda', 'openvino', 'tensorrt'], f'device is invalid: {device}'

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.mode = None
        self.session = None
        self.input_name = None
        self.request = None
        if os.path.splitext(model_path)[-1] == '.onnx':
            self.mode = 'onnx'
            if device == 'cpu':
                providers = ['CPUExecutionProvider']
            elif device == 'cuda':
                providers = ['CUDAExecutionProvider']
            elif device == 'openvino':
                providers = ['OpenVINOExecutionProvider']
            elif device == 'tensorrt':
                providers = ['TensorrtExecutionProvider']
            else:
                raise ValueError(f'device is invalid: {device}')
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
            session_input = self.session.get_inputs()[0]
            assert session_input.shape[2] == session_input.shape[3], 'The input shape must be square.'
            self.img_size = session_input.shape[2]
            self.input_name = session_input.name

        elif os.path.splitext(model_path)[-1] == '.xml':
            self.mode = 'openvino'
            import openvino.runtime
            core = openvino.runtime.Core()
            compiled_model = core.compile_model(model_path, 'CPU')
            self.request = compiled_model.create_infer_request()
            input_shape = self.request.inputs[0].shape
            assert input_shape[2] == input_shape[3], 'The input shape must be square.'
            self.img_size = input_shape[2]
        else:
            raise ValueError(f'Wrong file extension: {os.path.splitext(model_path)[-1]}')

    def _transform_image(self, img: np.ndarray, scale_ratio=1.0) -> Tuple[np.ndarray, float]:
        """
        Resizes the input image to fit img_size while preserving aspect ratio.
        (BGR to RGB, HWC to CHW, 0~1 normalization, and adding batch dimension)
        """
        img, scale = resize_preserving_aspect_ratio(img, self.img_size, scale_ratio)

        pad = (0, self.img_size - img.shape[0], 0, self.img_size - img.shape[1])
        img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # HWC to BCHW, BGR to RGB, uint8 to fp32
        img = np.ascontiguousarray(np.expand_dims(img.transpose((2, 0, 1))[::-1], 0), np.float32)
        img /= 255  # 0~255 to 0~1
        return img, scale

    def _non_max_suppression(self, pred: np.ndarray) -> np.ndarray:
        # obj_conf
        pred = pred[0]
        pred = pred[pred[:, 4] > self.conf_thres]
        if not pred.shape[0]:
            return pred

        # Compute conf
        pred[:, 4] *= pred[:, 15]  # conf = obj_conf * cls_conf
        pred = pred[:, :15][pred[:, 4] > self.conf_thres]
        if not pred.shape[0]:
            return pred

        # Box (cx, cy, w, h) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # sort by confidence
        pred = pred[pred[:, 4].argsort()[::-1]]

        # NMS
        i = torchvision.ops.nms(torch.from_numpy(pred[:, :4]), torch.from_numpy(pred[:, 4]), self.iou_thres).tolist()

        pred = pred[i]
        return pred

    def _inference(self, img: np.ndarray) -> np.ndarray:
        if self.mode == 'onnx':
            pred = self.session.run(None, {self.input_name: img})[0]
        elif self.mode == 'openvino':
            pred = self.request.infer({0: img}).popitem()[1]
        else:
            raise ValueError(f'Wrong mode: {self.mode}')
        pred = self._non_max_suppression(pred)
        return pred

    def _padded_detect_one(self, img: np.ndarray) -> Optional[np.ndarray]:
        original_img_shape = img.shape[:2]
        transformed_img, scale = self._transform_image(img, scale_ratio=1.5)
        pred = self._inference(transformed_img)
        if pred.shape[0] > 0:
            # Rescale coordinates from inference size to input image size
            pred[:, :4] /= scale
            pred[:, 5:] /= scale
            pred = clip_coords(pred, original_img_shape)
            return pred
        else:
            return None

    def detect_one(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Perform face detection on a single image.
        Args:
            img: Input image read using OpenCV. (HWC, BGR)
        Return:
            pred:
                Post-processed prediction. Shape=(number of faces, 15)
                15 is composed of bbox coordinates(4), object confidence(1), and landmarks coordinates(10).
                The coordinate format is x1y1x2y2 (bbox), xy per point (landmarks).
                The unit is image pixel.
                If no face is detected, output None.
        """
        original_img_shape = img.shape[:2]
        transformed_img, scale = self._transform_image(img)
        pred = self._inference(transformed_img)
        if pred.shape[0] > 0:
            # Rescale coordinates from inference size to input image size
            pred[:, :4] /= scale
            pred[:, 5:] /= scale
            pred = clip_coords(pred, original_img_shape)
            return pred
        else:
            return self._padded_detect_one(img)

    def parse_prediction(self, pred: np.ndarray) -> Tuple[List, List, List]:
        """Parse prediction to bbox, confidence, and landmarks."""
        bbox = pred[:, :4].round().astype(np.int32).tolist()
        conf = pred[:, 4].tolist()
        landmarks = pred[:, 5:].round().astype(np.int32).tolist()
        return bbox, conf, landmarks
