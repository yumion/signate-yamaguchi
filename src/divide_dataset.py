from pathlib import Path
import shutil
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import GroupKFold

from convert_coco import load_annotation, save_annotation


def main():
    src_dir = Path('../input/train_image')
    dst_dir = Path('../input/train_5cv_3classes')
    annotation_filename = 'annotations_3classes.json'
    n_splits = 5

    data = get_dataframe(src_dir)
    train, test = data_split(data, n_splits=n_splits)
    save_cv(test, src_dir, dst_dir, annotation_filename, train_test='test', symlink=True)  # foldフォルダ直下に画像を全て保存
    save_cv(train, src_dir, dst_dir, annotation_filename, train_test='train', symlink=True)  # foldフォルダ直下に画像を全て保存
    # save_group(test, src_dir, dst_dir, symlink=True)  # 動画単位でフォルダ分けして保存


def get_dataframe(src_dir):
    out = {'video': [], 'image': []}
    for image_dir in src_dir.glob('scene*'):
        for image in image_dir.glob('images/*.png'):
            out['video'].append(image_dir.name)
            out['image'].append(image.name)
    return out


def data_split(data, n_splits):
    group_kf = GroupKFold(n_splits)
    x = np.array(data['image'])
    groups = np.array(data['video'])
    train, test = {}, {}
    for i, (train_idxes, test_idxes) in enumerate(group_kf.split(data['image'], groups=groups)):
        train[i] = {
            'group': groups[train_idxes].tolist(),
            'data': x[train_idxes].tolist()
        }
        test[i] = {
            'group': groups[test_idxes].tolist(),
            'data': x[test_idxes].tolist(),
        }
        print(f"Fold {i + 1}:")
        print(f"  Train: group={np.unique(groups[train_idxes])}")
        print(f"  Test: group={np.unique(groups[test_idxes])}")
    return train, test


def save_cv(folds: dict, src_dir: Path, dst_dir: Path, annotation_filename: str, train_test: str = 'test', symlink: bool = False):
    for i, fold in folds.items():
        save_image(fold, src_dir, dst_dir / f'cv{i + 1}/{train_test}', symlink)
        dst_json = concat_annotation(src_dir, list(set(fold['group'])), annotation_filename)
        save_annotation(dst_json, dst_dir / f'cv{i + 1}/{train_test}' / annotation_filename)


def save_image(fold: dict, src_dir: Path, dst_dir: Path, symlink: bool = False):
    for video, image in tqdm(zip(fold['group'], fold['data']), desc='image'):
        src_image_p = src_dir / video / 'images' / image
        dst_image_p = dst_dir / 'images' / image
        dst_image_p.parent.mkdir(parents=True, exist_ok=True)
        if symlink:
            dst_image_p.resolve().symlink_to(src_image_p.resolve())
        else:
            shutil.copytree(src_image_p, dst_image_p)


def concat_annotation(src_dir: Path, sub_dir_names: list, filename: str):
    dst_json = {'images': [], 'annotations': []}
    for video in tqdm(sub_dir_names, desc='annotation'):
        src_json = load_annotation(src_dir / video / filename)
        dst_json['images'].extend(src_json['images'])
        dst_json['annotations'].extend(src_json['annotations'])
        dst_json['categories'] = src_json['categories']
    return dst_json


def save_group(folds, src_dir, dst_dir, symlink=False):
    for i, fold in folds.items():
        groups = np.unique(fold['group'])
        for image_dir in tqdm(groups, desc=f'cv{i + 1}'):
            src_p = src_dir / image_dir
            dst_p = dst_dir / f'cv{i + 1}' / image_dir
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            if symlink:
                symlink_dir(src_p, dst_p)
            else:
                shutil.copytree(src_p, dst_p)


def symlink_dir(src_dir, dst_dir):
    # シンボリックリンクで保存するには、フォルダごとではなくファイル単位でリンクつけないと見れなくなる
    # src_dir以下のファイルを全てリンク貼る
    for p in tqdm(src_dir.glob('**/*.*')):
        # 絶対パスにしないと参照できない
        dst_child_p = (dst_dir / str(p).replace(str(src_dir) + '/', '')).resolve()
        dst_child_p.parent.mkdir(parents=True, exist_ok=True)
        if not dst_child_p.exists():
            dst_child_p.symlink_to(p.resolve())


if __name__ == '__main__':
    main()
