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
    TOTAL_EPOCHS=200,
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
TRAIN = dict(CT_START=0.2, CT_P_START=0.2) # we start cross task loss at maxiter*0.2

'''
without denormalization
ad_2     23.33  55.38      38.24   51.87  33.43   49.26    23.47   72.77   58.30  32.92        55.57  61.42  40.04   45.85
ad_5     59.05  91.95      81.76   89.47  75.75   89.79    57.75   97.65   95.37  74.79        91.83  95.11  80.45   83.13
ad_10    86.00  99.52      96.57   98.82  95.51   98.61    84.79   99.81   99.71  94.77        99.08  99.42  95.75   96.03
rete_2   75.43  80.12      81.86   84.65  73.25   76.31    73.80   85.63   65.15  76.02        72.73  85.51  69.59   76.93
rete_5   97.24  99.22      99.41   99.21  99.10   98.91    97.93   99.62   97.97  99.05        97.55  99.42  96.22   98.53
rete_10  99.33  99.90      99.80   99.90  99.90   99.80    99.81   99.81   99.81  99.62        99.49  99.71  99.15   99.70
re_2     75.81  80.70      82.45   84.74  73.55   78.00    74.46   86.01   66.31  76.50        72.93  86.08  70.54   77.55
re_5     97.33  99.22      99.41   99.21  99.10   99.31    97.93   99.62   97.97  99.05        97.55  99.42  96.22   98.57
re_10    99.43  99.90      99.80   99.90  99.90   99.80    99.81   99.81   99.81  99.62        99.49  99.71  99.24   99.71
te_2     96.76  98.45      97.84   98.62  98.80   97.13    96.71   98.50   96.43  97.91        97.96  98.08  95.18   97.57
te_5     99.24  100.00     99.80   99.90  99.90   99.41    99.91   99.81   99.81  99.71        99.80  99.71  99.15   99.70
te_10    99.71  100.00     99.90   99.90  100.00  99.90    100.00  99.91   99.90  99.90        99.90  99.71  99.91   99.90
proj_2   95.33  87.49      92.35   92.52  95.01   78.99    94.18   95.02   92.76  96.29        85.50  80.90  86.31   90.20
proj_5   98.48  99.13      99.31   99.61  99.50   99.11    99.15   99.15   99.32  99.52        98.57  98.18  98.11   99.01
proj_10  99.90  100.00     99.90   99.90  99.90   99.80    100.00  99.91   99.90  99.81        99.59  99.71  100.00  99.87
re       1.69   1.47       1.42    1.32   1.63    1.49     1.67    1.27    1.84   1.63         1.74   1.42   1.84    1.57
te       0.01   0.01       0.01    0.01   0.01    0.01     0.01    0.01    0.01   0.01         0.01   0.01   0.01    0.01

Process finished with exit code 0

'''