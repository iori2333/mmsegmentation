_base_ = ['./segformer_mit-b0_0.5x_160k_hsicity2hsi.py']

model = dict(
    backbone=dict(
        in_channels=128,
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b5.pth'),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
