from pathlib import Path
import cv2
import numpy as np

from mmdet.utils import register_all_modules as register_all_det_modules
from mmdet.apis import init_detector, inference_detector
# from mmyolo.utils import register_all_modules as register_all_yolo_modules
from mmcls.utils import register_all_modules as register_all_cls_modules
from mmcls.apis import init_model, inference_model
from abc_predictor import ScoringService

register_all_det_modules()
# register_all_yolo_modules()
register_all_cls_modules(False)

# from mmengine.registry import count_registered_modules
# print(count_registered_modules())
# import json
# with open('registry.json', 'w') as fw:
#     json.dump(count_registered_modules(), fw, indent=4)


class ScoringService(ScoringService):
    pred_score_thr = 0.3
    ensemble_method = 'AND'
    classes = [
        '要補修-1.区画線',
        '要補修-2.道路標識',
        '要補修-3.照明',
        '補修不要-1.区画線',
        '補修不要-2.道路標識',
        '補修不要-3.照明'
    ]

    @classmethod
    def get_model(cls, model_path):
        model_path = Path(model_path)
        # Build the model from a config file and a checkpoint file
        cls.det_model = init_detector(
            str(model_path / 'detection/config.py'),
            str(model_path / 'detection/checkpoint.pth'),
            device='cuda:0')
        cls.cls_models = {
            'line': init_model(
                str(model_path / 'classification/line/config.py'),
                str(model_path / 'classification/line/checkpoint.pth'),
                device='cuda:0'),
            'sign': init_model(
                str(model_path / 'classification/sign/config.py'),
                str(model_path / 'classification/sign/checkpoint.pth'),
                device='cuda:0'),
            'light': init_model(
                str(model_path / 'classification/light/config.py'),
                str(model_path / 'classification/light/checkpoint.pth'),
                device='cuda:0'),
        }
        return True

    @classmethod
    def predict(cls, input):
        """Predict method
        Args:
            input: Data of the sample you want to make inference from (str)
        Returns:
            list: Inference for the given input.
        """
        prediction = []
        cap = cv2.VideoCapture(input)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if ret:
                result = cls.inference(frame)
                result['frame_id'] = frame_id
                prediction.append(result)
                frame_id += 1
            else:
                break
        return prediction

    @classmethod
    def inference(cls, frame):
        bboxes, det_labels = cls.detect(frame)
        cls_labels = cls.classify(frame, bboxes, det_labels)
        labels = cls.ensemble(det_labels, cls_labels, cls.ensemble_method)
        # 画像中に白線、標識、街頭がそれぞれ要補修が1つでもあればフラグを立てて返す
        result = {'line': 0, 'sign': 0, 'light': 0}
        for label in labels:
            label_name = cls.classes[label]
            if label_name == '要補修-1.区画線':
                result['line'] = 1
            if label_name == '要補修-2.道路標識':
                result['sign'] = 1
            if label_name == '要補修-3.照明':
                result['light'] = 1
        return result

    @classmethod
    def detect(cls, frame: np.ndarray):
        """mmdet inferece method
        Args:
            frame (np.ndarray): BGR image.
        Return:
            result (dict): {'line': flag, 'sign': flag, 'light': flag}
        """
        # 推論
        data_sample = inference_detector(cls.det_model, frame)
        # 予測instanceを取得
        pred_instances = data_sample.pred_instances
        pred_instances = pred_instances[pred_instances.scores > cls.pred_score_thr]
        # instanceからbboxとlabel idを取得
        bboxes = pred_instances.bboxes
        labels = pred_instances.labels
        return bboxes.cpu().detach().numpy(), labels.cpu().detach().numpy()

    @classmethod
    def classify(cls, frame, bboxes, labels):
        result = []
        # inference per a bbox
        for bbox, label in zip(bboxes, labels):
            # switch classification model
            label_name = cls.classes[label]
            if '区画線' in label_name:
                classifier = cls.cls_models['line']
            elif '道路標識' in label_name:
                classifier = cls.cls_models['sign']
            elif '照明' in label_name:
                classifier = cls.cls_models['light']
            # crop frame as a bbox
            x1, y1, x2, y2 = cls._xywh2xyxy(bbox)
            crop = frame[y1:y2, x1:x2]
            # inference against bbox
            res = inference_model(classifier, crop)
            # convert class id from binary classfication to 6 classes detection
            pred_label = cls.classes.index(res['pred_class'])
            result.append(pred_label)
        return result

    @classmethod
    def ensemble(cls, det_labels, cls_labels, mode='AND'):
        """ensemble of detection class and classification results

        Args:
            det_labels (list): detection model class results per bbox
            cls_labels (list): each binrary classification model results per bbox
            mode (Literal['AND', 'OR'], optional): how to ensemble.
                                                   if AND, (0,1) -> 0, if OR, (0,1) -> 1.
                                                   Defaults to 'AND'.
        """
        det_labels = np.array(det_labels)
        cls_labels = np.array(cls_labels)
        # onehot encoding
        det_onehot = np.eye(len(cls.classes))[det_labels]
        cls_onehot = np.eye(len(cls.classes))[cls_labels]
        # ensemble
        if mode == 'AND':
            labels_onehot = det_onehot * cls_onehot
        elif mode == 'OR':
            labels_onehot = det_onehot + cls_onehot
        else:
            NotImplementedError
        # convert class id
        labels = np.argmax(labels_onehot, axis=-1)
        return labels

    @classmethod
    def _xywh2xyxy(cls, bbox: np.ndarray) -> list:
        _bbox = bbox.tolist()
        return [
            int(_bbox[0]),
            int(_bbox[1]),
            int(_bbox[0] + _bbox[2]),
            int(_bbox[1] + _bbox[3]),
        ]
