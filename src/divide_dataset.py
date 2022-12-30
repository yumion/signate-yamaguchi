from pathlib import Path
import shutil
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import GroupKFold


def main():
    src_dir = Path('../input/train_image')
    dst_dir = Path('../input/train_5cv')
    n_splits = 5

    data = get_dataframe(src_dir)
    folds = data_split(data, n_splits=n_splits)
    save_group(folds, src_dir, dst_dir, symlink=True)  # 動画単位でフォルダ分けして保存
    # save_data(folds, src_dir, dst_dir, symlink=True)  # foldフォルダ直下に画像を全て保存


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
    out = {}
    for i, (train_idxes, test_idxes) in enumerate(group_kf.split(data['image'], groups=groups)):
        out[i] = {
            'group': groups[test_idxes].tolist(),
            'data': x[test_idxes].tolist(),
        }
        print(f"Fold {i}:")
        print(f"  Train: group={np.unique(groups[train_idxes])}")
        print(f"  Test: group={np.unique(groups[test_idxes])}")
    return out


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


def save_data(folds, src_dir, dst_dir, symlink=False):
    for i, fold in folds.items():
        for video, image in tqdm(zip(fold['group'], fold['data']), desc=f'cv{i + 1}'):
            src_p = src_dir / video / 'images' / image
            dst_p = dst_dir / f'cv{i + 1}' / 'images' / image
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            if symlink:
                dst_p.resolve().symlink_to(src_p.resolve())
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
