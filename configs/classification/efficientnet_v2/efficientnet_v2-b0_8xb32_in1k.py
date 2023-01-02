_base_ = [
    '../cv1.py',
    '../schedule_20e.py',
    '../../../src/mmclassification/configs/_base_/models/efficientnet_v2/efficientnet_v2_b0.py',
    '../../../src/mmclassification/configs/_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=2))

load_from = "https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-b0_3rdparty_in1k_20221221-9ef6e736.pth"
