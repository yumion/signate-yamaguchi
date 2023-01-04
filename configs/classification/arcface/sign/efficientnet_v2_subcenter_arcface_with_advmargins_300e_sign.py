_base_ = [
    'cv1_sign.py',
    '../efficientnet_v2_subcenter_arcface_with_advmargins_300e.py',
]


train_dataloader = dict(
    batch_size=64,
    num_workers=16)

val_dataloader = dict(
    batch_size=64,
    num_workers=16)
