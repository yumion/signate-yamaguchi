# train, val, test setting
max_epochs = 300
warmup_epochs = 5

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
# test_cfg = dict()

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='RAdam',
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=True,
        begin=0,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs - warmup_epochs,
        by_epoch=True,
        begin=warmup_epochs,
        end=max_epochs,
    ),
    dict(
        type='CosineAnnealingParamScheduler',
        param_name='weight_decay',
        eta_min=0.00001,
        by_epoch=True,
        begin=0,
        end=max_epochs)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
