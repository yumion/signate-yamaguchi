# Only for evaluation
_base_ = [
    '../../../_base_/models/swin_transformer/base_224.py',
    '../../../_base_/datasets/sar_rarp50_bs32_256x512_randaug/cv1.py',
    '../../../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../../../_base_/default_runtime.py'
]

batch_size = 24
runner = dict(type='EpochBasedRunner', max_epochs=10)

# dataset settings
data = dict(samples_per_gpu=batch_size)

# model settings
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth'
model = dict(
    backbone=dict(
        img_size=(256, 512),
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),  # (H,W)
    head=dict(num_classes=8),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=8, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=8, prob=0.5)
    ]))

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(lr=5e-4, weight_decay=1e-5)
# learning policy
lr_config = dict(
    warmup_iters=1500,
    warmup_by_epoch=False)

# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=1, by_epoch=True)
