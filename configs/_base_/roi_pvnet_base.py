_base_ = "./common_base.py"
# -----------------------------------------------------------------------------
# base model cfg for roi_pvnet
# -----------------------------------------------------------------------------
MODEL = dict(
    DEVICE="cuda",
    WEIGHTS="",
    # PIXEL_MEAN = [103.530, 116.280, 123.675]  # bgr
    # PIXEL_STD = [57.375, 57.120, 58.395]
    # PIXEL_MEAN = [123.675, 116.280, 103.530]  # rgb
    # PIXEL_STD = [58.395, 57.120, 57.375]
    PIXEL_MEAN=[0, 0, 0],  # to [0,1]
    PIXEL_STD=[255.0, 255.0, 255.0],
    LOAD_DETS_TEST=False,
    POSE_NET=dict(
        NAME="RoI_PVNet",  # used module file name
        TASK="rot",
        NUM_CLASSES=13,
        USE_MTL=False,  # uncertainty multi-task weighting
        INPUT_RES=256,
        OUTPUT_RES=64,
        ## backbone
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/resnet34",
                in_chans=3,
                features_only=True,
                pretrained=True,
                out_indices=(4,),
            ),
        ),
        ## geo head: Mask, Vecmap
        GEO_HEAD=dict(
            FREEZE=False,
            LR_MULT=1.0,
            INIT_CFG=dict(
                type="TopDownMaskVecmapHead",
                in_dim=512,
                up_types=("deconv", "bilinear", "bilinear"),  # stride 32 to 4
                deconv_kernel_size=3,
                num_conv_per_block=2,
                feat_dim=256,
                feat_kernel_size=3,
                norm="BN",
                num_gn_groups=32,
                act="lrelu",  # relu | lrelu | silu (swish) | gelu | mish
                out_kernel_size=1,
                out_layer_shared=True,
            ),
            VECMAP_TYPE="fps8_and_center",
            VECMAP_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            MASK_THR_TEST=0.5,
        ),
        ## trans head
        TRANS_HEAD=dict(
            ENABLED=True,
            FREEZE=True,  # unfreeze to train trans head
            LR_MULT=1.0,
            INIT_CFG=dict(
                type="ConvFCTransHead",
                in_channels=512,
                num_conv_layers=3,
                conv_feat_dim=256,
                kernel_size=3,
                num_fc_layers=2,
                fc_feat_dim=4096,
            ),
            TRANS_CLASS_AWARE=False,
            TRANS_TYPE="centroid_z",  # trans | centroid_z
            Z_TYPE="REL",  # REL | ABS | LOG | NEG_LOG
        ),
        LOSS_CFG=dict(
            # vecmap loss ----------------------------
            VECMAP_LOSS_TYPE="L1",  # L1 | CE_coor
            VECMAP_LOSS_MASK_GT="visib",  # trunc | visib | obj
            VECMAP_LW=1.0,
            # mask loss ---------------------------
            MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            # centroid loss -----------------------
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=0.0,
            Z_LOSS_TYPE="L1",
            Z_LW=0.0,
            TRANS_LOSS_TYPE="L1",
            TRANS_LW=0.0,
        ),
    ),
    # some d2 keys but not used
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False,
)

TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    TEST_BBOX_TYPE="est",  # gt | est
    PNP_TYPE="iter_pnp",  # upnp | upnp_v2 | iter_pnp | epnp
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
)
