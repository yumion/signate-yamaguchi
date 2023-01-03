# dataset settings
dataset_type = 'SARRARP50'
img_norm_cfg = dict(
    mean=[0.32858519781689005 * 255, 0.15265839395622285 * 255, 0.14655234887549404 * 255],  # (R,G,B)
    std=[0.07691241763785549 * 255, 0.053818967599625046 * 255, 0.056615884572508365],  # (R,G,B)
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Albu',
         transforms=[
             dict(type='HorizontalFlip', p=0.5),
             dict(type='ShiftScaleRotate',
                  border_mode=0,
                  rotate_limit=30,
                  scale_limit=0.5,
                  shift_limit=0.25,
                  p=0.5),
             dict(type='Resize',
                  height=256,
                  width=512,
                  interpolation=4)  # cv2.INTER_LANCZOS4
         ],
         keymap=dict(img='image'),
         update_pad_shape=True),
    dict(type='Normalize', **img_norm_cfg),  # only BGR to RGB
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Albu',
         transforms=[
             dict(type='Resize',
                  height=256,
                  width=512,
                  interpolation=4)  # cv2.INTER_LANCZOS4
         ],
         keymap=dict(img='image'),
         update_pad_shape=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_prefix='/mnt/data1/input/SAR-RARP50/20220723/cv1/train',
        image_dir='rgb',
        ann_file='action_discrete.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/mnt/data1/input/SAR-RARP50/20220723/cv1/valid',
        image_dir='rgb',
        ann_file='action_discrete.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/mnt/data1/input/SAR-RARP50/20220723/cv1/valid',
        image_dir='rgb',
        ann_file='action_discrete.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy', save_best='accuracy_top-1')
