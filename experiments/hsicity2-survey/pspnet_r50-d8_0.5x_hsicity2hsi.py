_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/hsicity2-hsi.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    backbone=dict(in_channels=128)
)