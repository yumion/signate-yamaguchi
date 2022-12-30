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
    save_dataset(folds, src_dir, dst_dir, symlink=True)


def get_dataframe(src_dir):
    out = {'video': [], 'image': []}
    for image_dir in src_dir.glob('scene*'):
        for image in image_dir.glob('images/*.png'):
            out['video'].append(image_dir.name)
            out['image'].append(image.stem)
    return out


def data_split(data, n_splits):
    group_kf = GroupKFold(n_splits)
    x = np.array(data['image'])
    groups = np.array(data['video'])
    out = {}
    for i, (train_idxes, test_idxes) in enumerate(group_kf.split(data['image'], groups=groups)):
        out[i] = {
            'group': np.unique(groups[test_idxes]).tolist(),
            'x': x[test_idxes].tolist(),
        }
        print(f"Fold {i}:")
        print(f"  Train: group={np.unique(groups[train_idxes])}")
        print(f"  Test: group={np.unique(groups[test_idxes])}")
    return out


def save_dataset(folds, src_dir, dst_dir, symlink=False):
    for i, fold in folds.items():
        for image_dir in tqdm(fold['group'], desc=f'cv{i + 1}'):
            src_p = src_dir / image_dir
            dst_p = dst_dir / f'cv{i + 1}' / image_dir
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            if symlink:
                # シンボリックリンクで保存するには、フォルダごとではなくファイル単位でリンクつけないと見れなくなる
                # src_p以下のファイルを全てリンク貼る
                for p in tqdm(src_p.glob('**/*.*')):
                    # 絶対パスにしないと参照できない
                    dst_child_p = (dst_p / str(p).replace(str(src_p) + '/', '')).resolve()
                    dst_child_p.parent.mkdir(parents=True, exist_ok=True)
                    if not dst_child_p.exists():
                        dst_child_p.symlink_to(p.resolve())
            else:
                shutil.copytree(src_p, dst_p)


if __name__ == '__main__':
    main()
