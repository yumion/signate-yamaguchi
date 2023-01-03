# dataset settings
dataset_type = 'CustomDataset'
data_root = 'input/train_5cv/cv1/'

data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[88.031, 95.264, 87.722],  # R,G,B
    std=[46.828, 51.956, 54.864],  # R,G,B
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=192,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=224, crop_padding=0),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='train/categories/light',
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='test/categories/light',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='SingleLabelMetric')

# test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
