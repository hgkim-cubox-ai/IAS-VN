import os
from typing import Tuple, Optional

import cv2
import numpy as np
import onnxruntime
import torch

from backbone_code.detect_segment_utils.general import resize_preserving_aspect_ratio, non_max_suppression, process_mask


class IDCardSegment:
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
        self.img_size = None
        self.input_name = None
        self.request = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
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

        elif os.path.splitext(model_path)[-1] == '.tflite':
            self.mode = 'tflite'
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            self.interpreter = Interpreter(model_path)  # load TFLite model
            self.interpreter.allocate_tensors()  # allocate
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            assert len(self.output_details) == 2, f'Wrong number of output: {len(self.output_details)}'
            assert self.input_details[0]['shape'][1] == self.input_details[0]['shape'][2], 'The input shape must be square.'
            self.img_size = self.input_details[0]['shape'][1]
        else:
            raise ValueError(f'Wrong file extension: {os.path.splitext(model_path)[-1]}')

    def _transform_image(self, img: np.ndarray, scale_ratio=1.0) -> Tuple[np.ndarray, float]:
        """
        Resizes the input image to fit img_size while preserving aspect ratio.
        (HWC to CHW, BGR to RGB, 0~1 normalization, and adding batch dimension)
        """
        img, scale = resize_preserving_aspect_ratio(img, self.img_size, scale_ratio)

        pad = (0, self.img_size - img.shape[0], 0, self.img_size - img.shape[1])
        img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # HWC to BCHW, BGR to RGB, uint8 to fp32
        img = np.ascontiguousarray(np.expand_dims(img.transpose((2, 0, 1))[::-1], 0), np.float32)
        img /= 255  # 0~255 to 0~1
        return img, scale

    def _inference(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == 'onnx':
            pred, proto = self.session.run(None, {self.input_name: img})
        elif self.mode == 'openvino':
            output = self.request.infer({0: img})
            proto = output.popitem()[1]
            pred = output.popitem()[1]
        elif self.mode == 'tflite':
            img = np.ascontiguousarray(img.transpose((0, 2, 3, 1)))
            int8 = self.input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
            if int8:
                scale, zero_point = self.input_details[0]['quantization']
                img = (img / scale + zero_point).astype(np.uint8)  # de-scale
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            y = []
            for output in self.output_details:
                x = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                y.append(x)
            y[0][..., :4] *= self.img_size  # xywh normalized to pixels
            pred, proto = y
        else:
            raise ValueError(f'Wrong mode: {self.mode}')
        pred = non_max_suppression(torch.from_numpy(pred), self.conf_thres, self.iou_thres, nm=32)[0].numpy()
        return pred, proto

    def segment_one(self, img: np.ndarray) -> Optional[Tuple[list, list, list, np.ndarray]]:
        """
        Perform idcard segmentation on a single image.
        Args:
            img: Input image read using OpenCV. (HWC, BGR)
        Return:
            pred:
                Post-processed prediction. (bbox, conf, masks)
                The bbox coordinate format is x1y1x2y2.
                The unit is image pixel.
                If no idcard is detected, output None.
        """
        original_img_shape = img.shape[:2]
        img, scale = self._transform_image(img)
        pred, proto = self._inference(img)
        if pred.shape[0] > 0:
            # Process mask
            masks = process_mask(proto[0], pred[:, 6:], pred[:, :4], (self.img_size, self.img_size), upsample=True)
            masks = np.array([cv2.resize(mask, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_NEAREST_EXACT)
                              for mask in masks])
            masks = masks[:, :original_img_shape[0], :original_img_shape[1]]

            # Rescale bboxes from inference size to input image size
            pred[:, :4] /= scale
            pred[:, [0, 2]] = pred[:, [0, 2]].clip(0, original_img_shape[1])  # x1, x2
            pred[:, [1, 3]] = pred[:, [1, 3]].clip(0, original_img_shape[0])  # y1, y2

            # Parse bbox and confidence
            bbox = pred[:, :4].round().astype(np.int32).tolist()
            conf = pred[:, 4].tolist()
            cls = pred[:, 5].tolist()
            return bbox, conf, cls, masks
        else:
            return None
