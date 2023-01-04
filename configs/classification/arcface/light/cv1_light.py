# dataset settings
dataset_type = 'CustomDataset'
data_root = 'input/train_5cv/cv1/'
category = 'light'

data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[88.031, 95.264, 87.722],  # R,G,B
    std=[46.828, 51.956, 54.864],  # R,G,B
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=192, crop_padding=0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=224, crop_padding=0),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=f'train/categories/{category}',
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=f'test/categories/{category}',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='SingleLabelMetric')

test_dataloader = val_dataloader
test_evaluator = val_evaluator
