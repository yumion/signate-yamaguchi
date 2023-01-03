# defaults to use registries in mmcls
default_scope = 'mmcls'

# configure default hooks
watch_metrics = ['single-label/precision', 'single-label/recall', 'single-label/f1-score']
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=1,
        save_best=watch_metrics, greater_keys=watch_metrics),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ClsVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
