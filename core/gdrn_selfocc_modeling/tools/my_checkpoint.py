import pickle
import os
from fvcore.common.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict, _process_mmcls_checkpoint


class MyCheckpointer(DetectionCheckpointer):
    """https://github.com/aim-
    uofa/AdelaiDet/blob/master/adet/checkpoint/adet_checkpoint.py Same as
    :class:`DetectronCheckpointer`, but is able to convert models in AdelaiDet,
    such as LPF backbone."""

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                if "weight_order" in data:
                    del data["weight_order"]
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        if filename.startswith("torchvision://") or filename.startswith(("http://", "https://")):
            loaded = _load_checkpoint(filename)  # load torchvision pretrained model using mmcv
        else:
            loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}

        basename = os.path.basename(filename).lower()
        if "lpf" in basename or "dla" in basename:
            loaded["matching_heuristics"] = True
        return loaded


def load_mmcls_ckpt(model, filename, map_location=None, strict=False, logger=None):
    ckpt = _load_checkpoint(filename, map_location=map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    ckpt = _process_mmcls_checkpoint(ckpt)

    # get state_dict from checkpoint
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in ckpt["state_dict"].items()}
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
