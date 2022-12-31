from tqdm import tqdm
import collections
from pathlib import Path
from convert_coco import load_annotation, save_annotation, CATEGORY2ID


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
        c = {"scene": json_p.stem} | {cat: 0 for cat in CATEGORY2ID.keys()}
        c.update(collections.Counter(scene_cats))
        counter_summary.append(c)

    save_annotation(counter_summary, Path("obj_count.json"))


def frame_anno_counter():
    # frameごとの最終的な正解データ（カテゴリごとに1つ以上要補修物体があれば要補修になる）のカウント
    counter_summary = []
    json_dir = Path('../input/train')
    for json_p in tqdm(sorted(json_dir.glob('scene_*.json'))):
        annos = load_annotation(json_p)
        c = {"scene": json_p.stem} | {cat: 0 for cat in CATEGORY2ID.keys()}
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

    save_annotation(counter_summary, Path("frame_anno_count.json"))


if __name__ == "__main__":
    frame_anno_counter()
