import numpy as np
import mmcv
import torch
if __name__ == "__main__":
    '''
    test_path = "/data/wanggu/Storage/BOP_DATASETS/ycbv/train_pbr/xyz_crop/000000/000015_000005-xyz.pkl"
    xyz = mmcv.load(test_path)
    t=1
    '''
    from core.gdrn_selfocc_modeling.tools.ycbv.generate_real_Q_gpu import run_real_one
    from core.gdrn_selfocc_modeling.tools.ycbv.generate_pbr_Q_gpu import run_this_one
    run_real_one()
    run_this_one()