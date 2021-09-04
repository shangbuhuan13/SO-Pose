_base_ = ["../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn_selfocc/lm/mm_r50v1d_a6_cPnP_GN_gelu_lm13"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        "Sometimes(0.4, CoarseDropout( p=0.1, size_percent=0.05) ),"
        # "Sometimes(0.5, Affine(scale=(1.0, 1.2))),"
        "Sometimes(0.5, GaussianBlur(np.random.rand())),"
        "Sometimes(0.5, Add((-20, 20), per_channel=0.3)),"
        "Sometimes(0.4, Invert(0.20, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),"
        "Sometimes(0.5, Multiply((0.7, 1.4))),"
        "Sometimes(0.5, LinearContrast((0.5, 2.0), per_channel=0.3))"
        "], random_order=False)"
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=24,
    TOTAL_EPOCHS=240,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
)

DATASETS = dict(
    TRAIN=(
        "lm_13_train",
        "lm_imgn_13_train_1k_per_obj",
    ),
    # TRAIN2=("lm_imgn_13_train_1k_per_obj",),
    # TRAIN2_RATIO=0.75,
    TEST=("lm_13_test",),
    DET_FILES_TEST=("/home/yan/gdnr_selfocc/datasets/BOP_DATASETS/lm/test/test_bboxes/bbox_faster_all.json",),)

MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    POSE_NET=dict(
        NAME="GDRN",
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="mmcls://resnet50_v1d",
            INIT_CFG=dict(
                _delete_=True,
                type="mm/ResNetV1d",
                depth=50,
                in_channels=3,
                out_indices=(3,),
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
            XYZ_LOSS_MASK_GT="visib",  # trunc | visib | obj
            XYZ_LW=1.0,
            # mask loss ---------------------------
            MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            # region loss -------------------------
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="visib",  # trunc | visib | obj
            REGION_LW=1.0,
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
        ),
    ),
)

TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="est")  # gt | est
TRAIN = dict(CT_START=0.6, CT_P_START=0.8) # we start cross task loss at maxiter*0.6

'''
with denormalization
objects  ape     benchvise  camera  can     cat     driller  duck    eggbox  glue    holepuncher  iron   lamp   phone  Avg(13)
ad_2     20.95   57.23      40.10   48.52   34.43   50.55    22.54   77.28   57.24   33.30        53.52  63.24  35.69  45.74
ad_5     55.52   92.43      82.84   89.47   76.45   89.30    58.31   97.84   96.24   75.07        91.22  94.82  79.60  83.01
ad_10    85.43   99.42      96.67   98.62   95.01   98.41    85.73   99.91   99.61   94.77        98.67  99.14  95.28  95.90
rete_2   73.14   78.56      82.35   83.96   73.05   77.40    73.62   87.70   65.73   74.31        73.03  85.70  68.37  76.69
rete_5   97.71   99.32      99.41   99.21   98.70   98.91    97.37   99.34   98.55   98.67        97.75  99.04  96.88  98.53
rete_10  99.81   99.90      99.90   99.80   99.80   99.90    99.72   99.81   99.90   99.62        99.49  99.71  98.96  99.72
re_2     74.10   78.86      82.75   84.06   73.65   78.00    74.46   87.89   66.99   74.60        73.85  85.99  69.69  77.30
re_5     97.81   99.32      99.41   99.21   98.70   99.21    97.46   99.34   98.55   98.67        97.75  99.04  96.88  98.57
re_10    99.81   99.90      99.90   99.80   99.80   99.90    99.72   99.81   100.00  99.62        99.49  99.71  98.96  99.73
te_2     96.76   98.84      98.43   98.62   98.10   97.42    96.34   98.59   95.95   97.62        96.73  98.37  95.09  97.45
te_5     99.24   100.00     99.80   99.90   100.00  99.41    99.81   99.91   99.81   99.71        99.80  99.71  99.34  99.73
te_10    99.90   100.00     99.90   99.90   100.00  100.00   100.00  99.91   99.90   100.00       99.80  99.71  99.91  99.92
proj_2   95.14   86.61      91.86   92.52   95.41   79.29    93.99   95.68   92.95   95.62        84.68  81.77  86.78  90.18
proj_5   98.57   99.42      99.51   99.51   99.70   99.01    99.25   99.15   99.32   99.43        98.67  98.18  98.58  99.10
proj_10  100.00  100.00     99.90   100.00  99.90   99.90    100.00  99.91   99.90   99.90        99.59  99.71  99.91  99.89
re       1.69    1.48       1.44    1.36    1.68    1.50     1.70    1.28    1.81    1.67         1.71   1.41   1.88   1.59
te       0.01    0.01       0.01    0.01    0.01    0.01     0.01    0.00    0.01    0.01         0.01   0.01   0.01   0.01


'''

