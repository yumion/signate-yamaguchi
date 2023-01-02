from tqdm import tqdm
import collections
from pathlib import Path
import cv2
import numpy as np
from convert_coco import load_annotation, save_annotation

# original classes
CATEGORIES = ['要補修-1.区画線', '要補修-2.道路標識', '要補修-3.照明',
              '補修不要-1.区画線', '補修不要-2.道路標識', '補修不要-3.照明']


def obj_counter():
    # sceneごとのobjectの数
    counter_summary = []
    json_dir = Path('../input/train')
    for json_p in tqdm(sorted(json_dir.glob('scene_*.json'))):
        scene_cats = []
        annos = load_annotation(json_p)
        for anno in annos:
            cats = list(anno['labels'].keys())
            scene_cats.extend(cats)
        c = {"scene": json_p.stem} | {cat: 0 for cat in CATEGORIES}
        c.update(collections.Counter(scene_cats))
        counter_summary.append(c)

    save_annotation(counter_summary, Path("../input/obj_count.json"))


def frame_anno_counter():
    # frameごとの最終的な正解データ（カテゴリごとに1つ以上要補修物体があれば要補修になる）のカウント
    counter_summary = []
    json_dir = Path('../input/train')
    for json_p in tqdm(sorted(json_dir.glob('scene_*.json'))):
        annos = load_annotation(json_p)
        c = {"scene": json_p.stem} | {cat: 0 for cat in CATEGORIES}
        for frame_anno in annos:

            if "要補修-1.区画線" in list(frame_anno['labels'].keys()):
                c["要補修-1.区画線"] += 1
            else:
                c["補修不要-1.区画線"] += 1

            if "要補修-2.道路標識" in list(frame_anno['labels'].keys()):
                c["要補修-2.道路標識"] += 1
            else:
                c["補修不要-2.道路標識"] += 1

            if "要補修-3.照明" in list(frame_anno['labels'].keys()):
                c["要補修-3.照明"] += 1
            else:
                c["補修不要-3.照明"] += 1

        counter_summary.append(c)

    save_annotation(counter_summary, Path("../input/frame_anno_count.json"))


def dataset_mean_std():
    total_sum = np.zeros(3)
    total_sum_square = np.zeros(3)
    img_dir = Path('../input/train_image')

    for img_file in img_dir.glob("*/images/*.png"):
        img = cv2.imread(str(img_file)).astype(np.float32)
        total_sum += np.sum(img, axis=(0, 1))
        total_sum_square += np.sum(img ** 2, axis=(0, 1))

    count = len(list(img_dir.glob("*/images/*.png"))) * 1920 * 1080
    total_mean = total_sum / count
    total_var = (total_sum_square / count) - (total_mean ** 2)
    total_std = np.sqrt(total_var)

    print('mean: ', str(total_mean))
    print('std:  ', str(total_std))

    # mean:  [87.72285137 95.26489529 88.03187912]
    # std:   [54.86452982 51.95639627 46.8286899 ]


def crop_bbox():
    # sceneごとカテゴリごとにbboxを切り出して保存する
    # train_image/scene_00/categories/補修不要-1.区画線/scene_00_000000_1.png

    img_dir = Path('../input/train_image')
    json_dir = Path('../input/train')

    for json_p in tqdm(sorted(json_dir.glob('scene_*.json'))):
        annos = load_annotation(json_p)
        scene = json_p.stem
        scene_img_dir = img_dir / scene / "images"

        for cat in CATEGORIES:
            cat_dir = img_dir / scene / "categories" / cat
            cat_dir.mkdir(parents=True, exist_ok=True)

        for frame_anno in annos:
            frame_id = frame_anno["frame_id"]
            img_path = scene_img_dir / f"{scene}_{frame_id:06d}.png"
            img = cv2.imread(str(img_path))
            for cat, bboxes in frame_anno['labels'].items():
                cat_dir = img_dir / scene / "categories" / cat
                for i, bbox in enumerate(bboxes, 1):
                    (x1, y1), (x2, y2) = bbox
                    x1 = x1 if x1 >= 0 else 0
                    y1 = y1 if y1 >= 0 else 0
                    rect = img[y1:y2, x1:x2]
                    save_file = cat_dir / f"{img_path.stem}_{i}.png"
                    cv2.imwrite(str(save_file), rect)


if __name__ == "__main__":
    crop_bbox()
