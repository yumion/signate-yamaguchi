_base_ = [
    '../schedule_coslr_coswd_300e.py',
    '../../../src/mmclassification/configs/_base_/models/efficientnet_v2/efficientnetv2_m.py',
    '../mmcls_runtime.py',
]

load_from = "https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_in21k-pre-3rdparty_in1k_20221220-a1013a04.pth"

model = dict(
    head=dict(
        _delete_=True,
        type='ArcFaceClsHead',
        num_classes=2,
        in_channels=1280,
        num_subcenters=3,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=None)
)

custom_hooks = [dict(type='SetAdaptiveMarginsHook')]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10))
