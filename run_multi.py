import sys
import argparse
import os
import json
import pandas as pd
import numpy as np


def make_flag(scene_id, annotations):
    flags = []
    for annotation in sorted(annotations, key=lambda x: x['frame_id']):
        frame_id = annotation['frame_id']
        line = 0
        sign = 0
        light = 0
        for label in annotation['labels']:
            if label == '要補修-1.区画線':
                line = 1
            elif label == '要補修-2.道路標識':
                sign = 1
            elif label == '要補修-3.照明':
                light = 1
        flags.append({'frame_id': frame_id, 'line': line, 'sign': sign, 'light': light})
    flags = pd.DataFrame(flags)
    flags['scene_id'] = [scene_id] * len(flags)

    return flags


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec-path', help='/path/to/submit/src')
    parser.add_argument('--data-dir', help='/path/to/train')
    parser.add_argument('--scene-ids', nargs='+', help='scene_id', default=['00'])
    args = parser.parse_args()

    return args


def main():
    # parse the arguments
    args = parse_args()
    exec_path = os.path.abspath(args.exec_path)
    data_dir = os.path.abspath(args.data_dir)

    # change the working directory
    os.chdir(exec_path)
    cwd = os.getcwd()
    print('\nMoved to {}'.format(cwd))
    model_path = os.path.join('..', 'model')

    # load the model
    sys.path.append(cwd)
    from mmdet_predictor import ScoringService

    print('\nLoading the model...', end='\r')
    model_flag = ScoringService.get_model(model_path)
    if model_flag:
        print('Loaded the model.   ')
    else:
        print('Could not load the model.')
        return None

    metrics = []
    for scene_id in args.scene_ids:
        # load the input data
        test_video_path = os.path.join(data_dir, 'scene_{}.mp4'.format(scene_id))
        test_annotation_path = os.path.join(data_dir, 'scene_{}.json'.format(scene_id))

        # run all and save the result
        print('\nPrediction for scene id: {}'.format(scene_id))
        with open(test_annotation_path, encoding='utf-8') as f:
            annotations = json.load(f)
        flags = make_flag(scene_id, annotations)
        prediction = ScoringService.predict(test_video_path)
        if not isinstance(prediction, list):
            print('Invalid data type. Must be list.')
            return None
        scene_ids = [scene_id] * len(prediction)

        prediction = pd.DataFrame(prediction)
        columns = set(prediction.columns)
        if columns != {'frame_id', 'line', 'sign', 'light'}:
            print('Invalid data name: {},  Excepted name: {}'.format(columns, {'frame_id', 'line', 'sign', 'light'}))
            return None

        for c in {'line', 'sign', 'light'}:
            values = set(prediction[c].unique())
            if not values.issubset({0, 1}):
                print('Invalid value found in {}. Must be 0 or 1'.format(c))
                return None
        prediction['scene_id'] = scene_ids
        merged = pd.merge(prediction, flags, on=('scene_id', 'frame_id'))

        mae = np.abs(merged[['line_x', 'sign_x', 'light_x']].values - merged[['line_y', 'sign_y', 'light_y']].values).mean()
        print('MAE: {}'.format(mae))
        metrics.append(mae)
    print(f'mean MAE (N={len(metrics)}): {np.mean(metrics)}')


if __name__ == '__main__':
    main()
