# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='R2Net',
        lo=True,
        tr=True
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=100.0, reduction='mean'))

train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=1)

train_dataset_type = 'SRFolderDataset'
val_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=False),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color'),

    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=False),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
data_root = './LOL_datasets'
scale = 1
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder=f'{data_root}/train/low',
            gt_folder=f'{data_root}/train/high',
            pipeline=train_pipeline,
            scale=scale,
            filename_tmpl='{}')),
    val=dict(
        type=val_dataset_type,
        lq_folder=f'{data_root}/test/low',
        gt_folder=f'{data_root}/test/high',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folder=f'{data_root}/test/low',
        gt_folder=f'{data_root}/test/high',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,  # The power of polynomial decay.
    min_lr=2e-4,  # The minimum learning rate to stable the training.
    by_epoch=False,  # Whethe count by epoch or not.)
)
evaluation = dict(interval=1000)
# checkpoint saving
checkpoint_config = dict(interval=10000, save_optimizer=False, by_epoch=False)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None
# runtime settings
total_iters = 10000
cudnn_benchmark = False
find_unused_parameters=True
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = f'./work_dirs_lol/{{ fileBasenameNoExtension }}'
