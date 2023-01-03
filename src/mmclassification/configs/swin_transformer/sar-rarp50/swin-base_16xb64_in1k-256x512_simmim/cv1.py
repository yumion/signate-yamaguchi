# Only for evaluation
_base_ = [
    '../../../_base_/models/swin_transformer/base_224.py',
    '../../../_base_/datasets/sar_rarp50_bs32_256x512_albu/cv1.py',
    '../../../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../../../_base_/default_runtime.py'
]

batch_size = 16
runner = dict(type='EpochBasedRunner', max_epochs=10)

# dataset settings
data = dict(
    samples_per_gpu=batch_size,
    train=dict(
        # data_prefix='/mnt/data1/input/SAR-RARP50/20220723/cv1/train'  # DLS
        data_prefix='/mnt/hdd2/data/SAR-RARP50/20220723/cv1/train'  # DL1
    ),
    val=dict(
        # data_prefix='/mnt/data1/input/SAR-RARP50/20220723/cv1/valid',  # DLS
        data_prefix='/mnt/hdd2/data/SAR-RARP50/20220723/cv1/valid'  # DL1
    ),
    test=dict(
        # data_prefix='/mnt/data1/input/SAR-RARP50/20220723/cv1/valid',  # DLS
        data_prefix='/mnt/hdd2/data/SAR-RARP50/20220723/cv1/valid'  # DL1
    ))


# model settings
checkpoint = '/mnt/cloudy_z/result/SAR-RARP50/selfsup/sar-rarp50/simmim_swin-base_1xb2048-coslr-100e_in1k_256x512/backbone_e100.pth'
model = dict(
    backbone=dict(
        img_size=(256, 512),  # (H,W)
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint,)),
    head=dict(num_classes=8),
    train_cfg=dict(_delete_=True))

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(lr=5e-4, weight_decay=1e-5)
# learning policy
lr_config = dict(
    warmup_iters=1500,
    warmup_by_epoch=False)

# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=1, by_epoch=True)
