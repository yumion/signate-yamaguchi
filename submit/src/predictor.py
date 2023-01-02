from pathlib import Path
import cv2
import numpy as np

from mmdet.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector


register_all_modules()


class ScoringService:
    pred_score_thr = 0.3
    classes = (
        '要補修-1.区画線',
        '要補修-2.道路標識',
        '要補修-3.照明',
        '補修不要-1.区画線',
        '補修不要-2.道路標識',
        '補修不要-3.照明'
    )

    @classmethod
    def get_model(cls, model_path):
        model_path = Path(model_path)
        # Build the model from a config file and a checkpoint file
        config_file = model_path / 'config.py'
        checkpoint_file = model_path / 'checkpoint.pth'
        cls.model = init_detector(
            str(config_file), str(checkpoint_file), device='cuda:0')
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
    def inference(cls, frame: np.ndarray):
        """mmdet inferece method
        Args:
            frame (np.ndarray): BGR image.
        Return:
            result (dict): {'line': flag, 'sign': flag, 'light': flag}
        """
        # 推論
        data_sample = inference_detector(cls.model, frame)
        # 予測instanceを取得
        pred_instances = data_sample.pred_instances
        pred_instances = pred_instances[pred_instances.scores > cls.pred_score_thr]
        # instanceからbboxとlabel idを取得
        bboxes = pred_instances.bboxes
        labels = pred_instances.labels

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
