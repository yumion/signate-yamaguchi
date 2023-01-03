_base_ = [
    # 'yolo_pipeline.py',
    '../../../src/mmdetection/configs/mmyolo/yolov7/yolov7_e-p6_syncbn_fast_8x16b-300e_coco.py',
]

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_e-p6_syncbn_fast_8x16b-300e_coco/yolov7_e-p6_syncbn_fast_8x16b-300e_coco_20221126_102636-34425033.pth'

num_classes = 6
max_epochs = 50

# dataset settings
data_root = 'input/train_5cv/cv1/'
annotation_filename = 'annotations.json'
classes = (
    '要補修-1.区画線',
    '要補修-2.道路標識',
    '要補修-3.照明',
    '補修不要-1.区画線',
    '補修不要-2.道路標識',
    '補修不要-3.照明'
)


train_dataloader = dict(
    batch_size=6,
    num_workers=16,
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=f'train/{annotation_filename}',
        data_prefix=dict(img='train/images/'),
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=f'test/{annotation_filename}',
        data_prefix=dict(img='test/images/'),
    ))

test_dataloader = val_dataloader
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),  # Can be accelerated
    ann_file=data_root + f'test/{annotation_filename}',
    metric='bbox',
    classwise=True,
    iou_thrs=0.5)
test_evaluator = val_evaluator

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1)

# model
num_det_layers = 3
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        loss_cls=dict(loss_weight=0.3 * (num_classes / 80 * 3 / num_det_layers)),
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)
