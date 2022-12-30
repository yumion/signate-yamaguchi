import json
from pathlib import Path
from tqdm import tqdm


HEIGHT = 1080
WIDTH = 1920
CATEGORIES = [
    {'id': 0, 'name': '要補修-1.区画線'},
    {'id': 1, 'name': '要補修-2.道路標識'},
    {'id': 2, 'name': '要補修-3.照明'},
    {'id': 3, 'name': '補修不要-1.区画線'},
    {'id': 4, 'name': '補修不要-2.道路標識'},
    {'id': 5, 'name': '補修不要-3.照明'},
]
CATEGORY2ID = {category['name']: category['id'] for category in CATEGORIES}


def main():

    json_dir = Path('../input/train')
    image_dir = Path('../input/train_image')
    save_dir = image_dir

    # make coco json file per video
    for json_p in tqdm(json_dir.glob('*.json')):
        anno = load_annotation(json_p)
        coco_dataset = {
            'images': [],
            'annotations': [],
        }
        instance_id = 0  # serial instance id throughout video
        for image_id, image_p in enumerate(image_dir.glob(f'{json_p.stem}/images/*.png')):
            # add image
            image_info = {
                'id': image_id,
                'file_name': image_p.name,
                'height': HEIGHT,
                'width': WIDTH,
            }
            coco_dataset['images'].append(image_info)
            # add annotation
            labels = anno[image_id]['labels']
            for cat, bboxes in labels.items():
                for bbox in bboxes:
                    (x1, y1), (x2, y2) = bbox
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    annotation_info = {
                        # 'segmentation': None,
                        'id': instance_id,
                        'image_id': image_id,
                        'bbox': [x, y, w, h],
                        'category_id': CATEGORY2ID[cat],
                        'area': w * h,
                        'iscrowd': 0,
                    }
                    coco_dataset['annotations'].append(annotation_info)
                    instance_id += 1

        coco_dataset['categories'] = CATEGORIES
        save_path = save_dir / json_p.stem / 'annotations.json'
        save_annotation(coco_dataset, save_path)


def load_annotation(anno_path):
    with open(anno_path, 'r', encoding='utf-8') as fr:
        anno = json.load(fr)
    return anno


def save_annotation(anno, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as fw:
        json.dump(anno, fw, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
