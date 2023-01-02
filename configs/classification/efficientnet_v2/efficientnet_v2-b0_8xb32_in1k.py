_base_ = [
    '../cv1.py',
    '../schedule_20e.py',
    '../../../src/mmclassification/configs/_base_/models/efficientnet_v2/efficientnet_v2_b0.py',
    '../../../src/mmclassification/configs/_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=2))
