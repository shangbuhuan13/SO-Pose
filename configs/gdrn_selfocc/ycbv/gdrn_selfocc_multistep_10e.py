# about 3 days
_base_ = ["../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn_selfocc/ycbv/pbr_13e"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=True,
    CHANGE_BG_PROB=0.5,
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        # Sometimes(0.5, PerspectiveTransform(0.05)),
        # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))"
        "], random_order = False)"
        # aae
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=24,
    TOTAL_EPOCHS=15,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    CHECKPOINT_PERIOD=1,
    AMP=dict(ENABLED=True),
)

DATASETS = dict(
    TRAIN=("ycbv_train_real", "ycbv_train_pbr"),
    TEST=("ycbv_test",),
    # # AP    AP50  AR    inf.time  (faster RCNN)
    # # 75.10 93.00 81.40 25.4ms
    # DET_FILES_TEST=(
    #     "datasets/BOP_DATASETS/ycbv/test/test_bboxes/faster_R50_FPN_AugCosyAAE_HalfAnchor_ycbv_real_pbr_8e_test_keyframe_480x640.json",
    # ),
    # AP     AP50   AR   inf.time  (fcos_V57)
    # 79.581 96.949 84.2 56.572ms
    DET_FILES_TEST=(
        "/home/yan/gdnr_selfocc/datasets/BOP_DATASETS/ycbv/test/test_bboxes/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_ycbv_real_pbr_8e_test_keyframe.json",
    ),
    SYM_OBJS=["024_bowl", "036_wood_block", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"],  # ycbv
)

'''
MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    CDPN=dict(
        ROT_HEAD=dict(
            FREEZE=False,
            ROT_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            XYZ_LW=1.0,
            REGION_CLASS_AWARE=False,
            NUM_REGIONS=64,
        ),
        PNP_NET=dict(
            R_ONLY=False,
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
            PM_NORM_BY_EXTENT=True,
            PM_R_ONLY=True,
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
        ),
        TRANS_HEAD=dict(ENABLED=False),
    ),

)
'''
DATALOADER = dict(
    # Number of data loading threads
    NUM_WORKERS=12,
    FILTER_VISIB_THR=0.2,
)


MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    POSE_NET=dict(
        NAME="GDRN",
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/resnest50d",
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(4,),
            ),
        ),
        ## geo head: Mask, XYZ, Region
        GEO_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="TopDownMaskXyzRegionHead",
                in_dim=2048,  # this is num out channels of backbone conv feature
            ),
            NUM_REGIONS=64,
        ),
        ## selfocchead  Q0 occmask(optional)
        SELFOCC_HEAD=dict(
            OCCMASK_AWARE=False,
            Q0_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            FREEZE=False,
            INIT_CFG=dict(
                type="ConvSelfoccHead",
                in_dim=2048,
                feat_dim=256,
                feat_kernel_size=3,
                norm="GN",
                num_gn_groups=32,
                act="GELU",  # relu | lrelu | silu (swish) | gelu | mish
                out_kernel_size=1,
                out_layer_shared=False,
                Q0_num_classes=1,
                mask_num_classes=1,
            ),
            MIN_Q0_REGION=20,
            LR_MULT=1.0,

            REGION_CLASS_AWARE=False,
            MASK_THR_TEST=0.5,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(norm="GN", act="gelu"),
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        LOSS_CFG=dict(
            # xyz loss ----------------------------
            XYZ_LOSS_TYPE="L1",  # L1 | CE_coor
            XYZ_LOSS_MASK_GT="erode",  # trunc | visib | obj  | erode
            XYZ_LW=1.0,
            # mask loss ---------------------------
            MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            # region loss -------------------------
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="erode",  # trunc | visib | obj |erode
            REGION_LW=0.01,
            # pm loss --------------
            PM_R_ONLY=True,  # only do R loss in PM
            PM_LW=1.0,
            # centroid loss -------
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            # z loss -----------
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
            # Q0 loss ---------------------
            Q0_LOSS_TYPE="L1",
            Q0_LOSS_MASK_GT="visib",  # computed from Q0
            Q0_LW=1.0,
            # cross-task loss -------------------
            CT_LW=10.0,
            CT_P_LW=1.0,
            # occlusion mask loss weight
            OCC_LW=0.0,
            PM_NORM_BY_EXTENT=True,
            PM_LOSS_SYM=True,
            # Q direction
            QD_LW=0.0,
            #
            HANDLE_SYM=True,
        ),
    ),
)

VAL = dict(
    DATASET_NAME="ycbvposecnn",
    SPLIT_TYPE="",
    SCRIPT_PATH="../../../lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="ycbv_test_targets_keyframe.json",  # 'lm_test_targets_bb8.json'
    ERROR_TYPES="AUCadd,AUCadi,AUCad,ad,ABSadd,ABSadi,ABSad",
    USE_BOP=True,  # whether to use bop toolkit
)


TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="est")  # gt | est
TRAIN = dict(CT_START=0.2, CT_P_START=0.2)  # we start cross task loss at maxiter*0.6

