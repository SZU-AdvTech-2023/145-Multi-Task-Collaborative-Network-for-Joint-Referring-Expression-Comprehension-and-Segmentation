import torch.nn as nn

from simrec.config import LazyCall
from simrec.models.mcn_fusion3 import MCN
from simrec.models.backbones import YOLOv8Backbone
from simrec.models.heads import MCNhead, MCNhead_anchor_free
from simrec.models.language_encoders.lstm_sa import LSTM_SA
from simrec.layers.fusion_layer import SimpleFusion, MultiScaleFusion, GaranAttention

model = LazyCall(MCN)(
    visual_backbone=LazyCall(YOLOv8Backbone)(
        base_channels=80,
        base_depth=3,
        deep_mul=0.5,
        phi='x',
        pretrained=False,
        pretrained_weight_path="yolov8_x_backbone_weights.pth",
        freeze_backbone=True,
        multi_scale_outputs=True,
    ),
    language_encoder=LazyCall(LSTM_SA)(
        depth=3,
        hidden_size=512,
        num_heads=8,
        ffn_size=2048,
        flat_glimpses=1,
        word_embed_size=300,
        dropout_rate=0.1,
        # language_encoder.pretrained_emb and language.token_size is meant to be set
        # before instantiating
        freeze_embedding=True,
        use_glove=True,
    ),
    multi_scale_manner=LazyCall(MultiScaleFusion)(
        v_planes=(640, 640, 640),
        hiden_planes=640,  # TODO 根据yolov8x要求的通道数进行更改
        scaled=True
    ),
    # fusion_manner=LazyCall(SimpleFusion)(
    #     v_planes=640,
    #     q_planes=512,
    #     out_planes=640,
    # ),
    fusion_manner=LazyCall(nn.ModuleList)(
        modules=[
            LazyCall(SimpleFusion)(v_planes=320, out_planes=640, q_planes=512),
            LazyCall(SimpleFusion)(v_planes=640, out_planes=640, q_planes=512),
            LazyCall(SimpleFusion)(v_planes=640, out_planes=640, q_planes=512),
        ]
    ),
    det_attention=LazyCall(GaranAttention)(
        d_q=512,
        d_v=640
    ),
    seg_attention=LazyCall(GaranAttention)(
        d_q=512,
        d_v=640
    ),
    head=LazyCall(MCNhead_anchor_free)(
        hidden_size=640,
        # anchors=[[137, 256], [248, 272], [386, 271]],
        # arch_mask=[[0, 1, 2]],
        # layer_no=0,
        # in_ch=512,
        # n_classes=0,
        # ignore_thre=0.5,
        label_smooth=0.0,
        num_classes=0,
        width=1.0,
        strides=[32, ],
        in_channels=[640, ],
        act="silu",
        depthwise=False,
    )
)
