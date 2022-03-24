_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/hsicity2-rgb.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    pretrained='work_dirs/fcn_hr48_512x1024_160k_cityscapes/iter_160000.pth',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))
