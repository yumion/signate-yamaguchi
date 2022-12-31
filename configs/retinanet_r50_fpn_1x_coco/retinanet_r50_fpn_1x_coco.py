_base_ = [
    '../cv1.py',
    '../schedule_20e.py',
    '../../src/mmdetection/configs/_base_/models/retinanet_r50_fpn.py',
    '../../src/mmdetection/configs/_base_/default_runtime.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_mstrain_3x_coco/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth'

# model
model = dict(bbox_head=dict(num_classes=6))

# dataset
train_dataloader = dict(batch_size=8, num_workers=16)
val_dataloader = dict(batch_size=1, num_workers=16)
