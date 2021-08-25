import logging
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from setproctitle import setproctitle
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from mmcv import Config
import cv2

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))
from core.utils.default_args_setup import my_default_argument_parser, my_default_setup
from core.utils.my_setup import setup_for_distributed
from core.utils.my_checkpoint import MyCheckpointer
from core.utils import my_comm as comm

from lib.utils.utils import iprint
from lib.utils.setup_logger import setup_my_logger
from lib.utils.time_utils import get_time_str
import ref

from core.gdrn_selfocc_modeling.datasets.dataset_factory import register_datasets_in_cfg
#from core.gdrn_selfocc_modeling.engine.engine_utils import get_renderer
from core.gdrn_selfocc_modeling.engine.engine import do_test, do_train
from core.gdrn_selfocc_modeling.models import GDRN, GDRN_no_region, GDRN_cls, GDRN_cls2reg  # noqa


logger = logging.getLogger("detectron2")


def setup(args):
    """Create configs and perform basic setups."""
    cfg = Config.fromfile(args.config_file)
    if args.opts is not None:
        cfg.merge_from_dict(args.opts)
    ############## pre-process some cfg options ######################
    # NOTE: check if need to set OUTPUT_DIR automatically
    if cfg.OUTPUT_DIR.lower() == "auto":
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_ROOT, osp.splitext(args.config_file)[0].split("configs/")[1])
        iprint(f"OUTPUT_DIR was automatically set to: {cfg.OUTPUT_DIR}")

    if cfg.get("EXP_NAME", "") == "":
        setproctitle("{}.{}".format(osp.splitext(osp.basename(args.config_file))[0], get_time_str()))
    else:
        setproctitle("{}.{}".format(cfg.EXP_NAME, get_time_str()))

    if cfg.SOLVER.AMP.ENABLED:
        if torch.cuda.get_device_capability() <= (6, 1):
            iprint("Disable AMP for older GPUs")
            cfg.SOLVER.AMP.ENABLED = False

    # NOTE: pop some unwanted configs in detectron2
    # ---------------------------------------------------------
    cfg.SOLVER.pop("STEPS", None)
    cfg.SOLVER.pop("MAX_ITER", None)
    bs_ref = cfg.SOLVER.get("REFERENCE_BS", cfg.SOLVER.IMS_PER_BATCH)  # nominal batch size
    if bs_ref <= cfg.SOLVER.IMS_PER_BATCH:
        bs_ref = cfg.SOLVER.REFERENCE_BS = cfg.SOLVER.IMS_PER_BATCH
        # default DDP implementation is slow for accumulation according to: https://pytorch.org/docs/stable/notes/ddp.html
        # all-reduce operation is carried out during loss.backward().
        # Thus, there would be redundant all-reduce communications in a accumulation procedure,
        # which means, the result is still right but the training speed gets slower.
        # TODO: If acceleration is needed, there is an implementation of allreduce_post_accumulation
        # in https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
        accumulate_iter = max(round(bs_ref / cfg.SOLVER.IMS_PER_BATCH), 1)  # accumulate loss before optimizing
    else:
        accumulate_iter = 1
    # NOTE: get optimizer from string cfg dict
    if cfg.SOLVER.OPTIMIZER_CFG != "":
        if isinstance(cfg.SOLVER.OPTIMIZER_CFG, str):
            optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
            cfg.SOLVER.OPTIMIZER_CFG = optim_cfg
        else:
            optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
        iprint("optimizer_cfg:", optim_cfg)
        cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
        cfg.SOLVER.BASE_LR = optim_cfg["lr"]
        cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
        cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
        if accumulate_iter > 1:
            if "weight_decay" in cfg.SOLVER.OPTIMIZER_CFG:
                cfg.SOLVER.OPTIMIZER_CFG["weight_decay"] *= (
                    cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref
                )  # scale weight_decay
    if accumulate_iter > 1:
        cfg.SOLVER.WEIGHT_DECAY *= cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref
    # -------------------------------------------------------------------------
    if cfg.get("DEBUG", False):
        iprint("DEBUG")
        args.num_gpus = 1
        args.num_machines = 1
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.TRAIN.PRINT_FREQ = 1
    # register datasets
    register_datasets_in_cfg(cfg)

    exp_id = "{}".format(osp.splitext(osp.basename(args.config_file))[0])

    if args.eval_only:
        if cfg.TEST.USE_PNP:
            # NOTE: need to keep _test at last
            exp_id += "{}_test".format(cfg.TEST.PNP_TYPE.upper())
        else:
            exp_id += "_test"
    cfg.EXP_ID = exp_id
    cfg.RESUME = args.resume
    ####################################
    if args.launcher != "none":
        comm.init_dist(args.launcher, **cfg.DIST_PARAMS)
    # cfg.freeze()
    my_default_setup(cfg, args)
    # Setup logger
    setup_my_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="core")
    setup_my_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="lib")
    setup_for_distributed(is_master=comm.is_main_process())
    return cfg


def main(args):
    cfg = setup(args)

    distributed = comm.get_world_size() > 1

    # get renderer ----------------------
    if cfg.MODEL.POSE_NET.XYZ_ONLINE and not args.eval_only:
        train_dset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        data_ref = ref.__dict__[train_dset_meta.ref_key]
        train_obj_names = train_dset_meta.objs
        if distributed:
            # pci bus number, avoid rendering on the same gpu
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(torch.cuda.device_count())))
            gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            render_gpu_id = int(gpu_ids[comm.get_local_rank()])  # local rank
        else:
            render_gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        renderer = None #get_renderer(cfg, data_ref, obj_names=train_obj_names, gpu_id=render_gpu_id)
    else:
        renderer = None

    logger.info(f"Used GDRN module name: {cfg.MODEL.POSE_NET.NAME}")
    model, optimizer = eval(cfg.MODEL.POSE_NET.NAME).build_model_optimizer(cfg, is_test=args.eval_only)
    logger.info("Model:\n{}".format(model))

    if args.eval_only:
        MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return do_test(cfg, model)

    if distributed and args.launcher not in ["hvd", "bps"]:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )

    do_train(cfg, args, model, optimizer, renderer=renderer, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    import resource
    import argparse
    from mmcv import DictAction
    # RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(500000, hard_limit)
    iprint("soft limit: ", soft_limit, "hard limit: ", hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

    # construct args
    parser = argparse.ArgumentParser(
        epilog=None
               or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="../../configs/gdrn_selfocc/lmo/gdrn_selfocc_multistep_40E_pbr_QDEF.py", metavar="FILE", help="path to config file")
    assert osp.exists("../../configs/gdrn_selfocc/lmo/gdrn_selfocc_multistep_40E_pbr_QDEF.py")
    parser.add_argument(
        "--resume", default=True, action="store_true", help="whether to attempt to resume from the checkpoint directory"
    )
    parser.add_argument("--eval-only", default=False, action="store_true", help="perform evaluation only")
    # distributed training launcher:
    # none: non-distributed or use the detectron2 default launcher
    # pytorch: use pytorch launcher
    # hvd: use horovod launcher
    # bps: use byteps launcher
    parser.add_argument("--launcher", choices=["none", "pytorch", "hvd", "bps"], default="none", help="job launcher")
    # pytorch dist options ======
    parser.add_argument("--local_rank", type=int, default=0)
    # hvd options ======
    parser.add_argument(
        "--fp16_allreduce", action="store_true", default=False, help="use fp16 compression during allreduce for hvd"
    )
    parser.add_argument("--use-adasum", action="store_true", default=False, help="use adasum algorithm to do reduction")
    # bps options ======
    parser.add_argument(
        "--fp16_pushpull", action="store_true", default=False, help="use fp16 compression during pushpull for bps"
    )
    # -------------
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--opts", nargs="+", action=DictAction, help="arguments in dict, modify config using command-line args"
    )

    args = parser.parse_args()
    iprint("Command Line Args:", args)
    comm.init_dist_env_variables(args)

    if args.eval_only:
        torch.multiprocessing.set_sharing_strategy("file_system")

    if args.launcher != "none":
        main(args)
    else:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
