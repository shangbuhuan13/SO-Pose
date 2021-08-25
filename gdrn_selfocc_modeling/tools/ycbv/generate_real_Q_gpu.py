import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from datetime import timedelta

import mmcv
import numpy as np
from tqdm import tqdm
import torch
cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../../../..")
sys.path.insert(0, PROJ_ROOT)
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.vis_utils.image import grid_show

idx2class = {
    1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
    2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
    3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
    4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
    5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
    6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
    7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
    8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
    9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
    10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
    11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
    12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
    13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
    14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
    15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
    16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
    17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
    18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
    19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
    20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
    21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
}

class2idx = {_name: _id for _id, _name in idx2class.items()}

classes = idx2class.values()
classes = sorted(classes)

# DEPTH_FACTOR = 10000.
IM_H = 480
IM_W = 640
near = 0.01
far = 6.5

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/ycbv/"))

cls_indexes = sorted(idx2class.keys())
cls_names = [idx2class[cls_idx] for cls_idx in cls_indexes]
model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/ycbv/models"))
model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in cls_indexes]
texture_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.png") for obj_id in cls_indexes]
# scenes = [i for i in range(0, 91 + 1)]
test_scenes = [i for i in range(48, 59 + 1)]
train_real_scenes = [i for i in range(0, 91 + 1) if i not in test_scenes]
train_synt_scenes = [i for i in range(0, 79 + 1)]



def transformer(P0, R, t):  # P0, n,3
    P = (np.matmul(R, P0.T)).T + t.reshape(1, 3)
    return P


def transformer_back(P, R, t):  # 计算P0=RTP-RTt  P, 3*1
    P0 = np.matmul(R.T, P) - np.matmul(R.T, t)
    return P0


# P0 n,3
def projector(P0, K, R, t):  # 计算相机投影， 将P0经过R， t变换再投影到图像上
    p = (np.matmul(K, P0.T)).T / P0[:, 2:]  # n,3
    p = p[:, 0:2] / p[:, 2:]
    return p


def generate_rec_list(ids, model_info_paths, scale=1000):  # from json
    model_info = mmcv.load(model_info_paths)
    REC_LIST = {}
    for obj_id in ids:
        id = str(obj_id)
        diameter = model_info[id]["diameter"]
        x_min, x_size = model_info[id]["min_x"], model_info[id]["size_x"]
        y_min, y_size = model_info[id]["min_y"], model_info[id]["size_y"]
        z_min, z_size = model_info[id]["min_z"], model_info[id]["size_z"]
        x_max = x_min + x_size
        y_max = y_min + y_size
        z_max = z_min + z_size
        x_min = x_min / scale
        x_max = x_max / scale
        y_min = y_min / scale
        y_max = y_max / scale
        z_min = z_min / scale
        z_max = z_max / scale
        REC_LIST[id] = {
            "xmin": x_min,
            "xmax": x_max,
            "ymin": y_min,
            "ymax": y_max,
            "zmin": z_min,
            "zmax": z_max,
            "d": diameter,
        }
    return REC_LIST


def normalize_to_01(img):
    if img.max() != img.min():
        return (img - img.min()) / (img.max() - img.min())
    else:
        return img


def get_emb_show(bbox_emb):
    show_emb = bbox_emb.copy()
    show_emb = normalize_to_01(show_emb)
    return show_emb


def get_time_delta(sec):
    """Humanize timedelta given in seconds."""
    if sec < 0:
        return "{:.3g} seconds".format(sec)
    delta_time_str = str(timedelta(seconds=sec))
    return delta_time_str


def test_in_box(point, xmin, xmax, ymin, ymax, zmin, zmax, R, t):
    # 要先将点坐标变换回去
    point = np.matmul(R.T, point) - np.matmul(R.T, t)
    # print(point)
    if xmin < point[0] < xmax and ymin < point[1] < ymax and zmin < point[2] < zmax:
        return 1, point
    else:
        return 0, 0


def test_in_box_cuda(Q_x_0, Q_x_1, Q_x_2, xmin, xmax, ymin, ymax,
                     zmin, zmax, R_cuda, t_cuda):
    # transform back
    RT = R_cuda.permute(1, 0)
    RTt = torch.matmul(RT, t_cuda.view(3, 1)).view(3)
    Q0_x_0 = RT[0, 0] * Q_x_0 + RT[0, 1] * Q_x_1 + RT[0, 2] * Q_x_2 - RTt[0]
    Q0_x_1 = RT[1, 0] * Q_x_0 + RT[1, 1] * Q_x_1 + RT[1, 2] * Q_x_2 - RTt[1]
    Q0_x_2 = RT[2, 0] * Q_x_0 + RT[2, 1] * Q_x_1 + RT[2, 2] * Q_x_2 - RTt[2]
    mask_1_r = Q0_x_0 < xmax
    mask_1_l = Q0_x_0 > xmin
    mask_2_r = Q0_x_1 < ymax
    mask_2_l = Q0_x_1 > ymin
    mask_3_r = Q0_x_2 < zmax
    mask_3_l = Q0_x_2 > zmin
    mask = (mask_1_l & mask_1_r & mask_2_l & mask_2_r & mask_3_l & mask_3_r).float()
    return mask, Q0_x_0, Q0_x_1, Q0_x_2


class XyzGen(object):
    def __init__(self, split="train", scene="all"):
        if split == "train_real":
            scene_ids = train_real_scenes
            data_root = osp.join(data_dir, "train_real")
        elif split == "train_synt":
            scene_ids = train_synt_scenes
            data_root = osp.join(data_dir, "train_synt")
        elif split == "test":
            scene_ids = test_scenes
            data_root = osp.join(data_dir, "test")
        else:
            raise ValueError(f"split {split} error")

        if split in ["train_real", "train_synt", "test", "train_pbr"]:
            self.xyz_root = osp.normpath(osp.join(f"/data/wanggu/Storage", f"BOP_DATASETS/ycbv/{split}/xyz_crop"))
            self.occ_root = osp.normpath(osp.join(f"/data/wanggu/Storage", f"BOP_DATASETS/ycbv/{split}/Q0"))
        if scene == "all":
            sel_scene_ids = scene_ids
        else:
            assert int(scene) in scene_ids, f"{scene} not in {scene_ids}"
            sel_scene_ids = [int(scene)]
        print("split: ", split, "selected scene ids: ", sel_scene_ids)
        self.split = split
        self.scene = scene
        self.sel_scene_ids = sel_scene_ids
        self.data_root = data_root
        self.renderer = None

    def main(self):
        split = self.split
        scene = self.scene  # "all" or a single scene
        sel_scene_ids = self.sel_scene_ids
        data_root = self.data_root

        n_x = torch.from_numpy(np.array([[1.], [0.], [0.]], dtype=np.float32)).cuda()  # Q0_yz
        n_y = torch.from_numpy(np.array([[0.], [1.], [0.]], dtype=np.float32)).cuda()  # Q0_xz
        n_z = torch.from_numpy(np.array([[0.], [0.], [1.]], dtype=np.float32)).cuda()  # Q0_xy

        model_info_path = osp.join(model_dir, "models_info.json")
        REC_LIST = generate_rec_list(cls_indexes, model_info_path)
        height = 480
        width = 640

        h_coor = torch.linspace(0, height - 1, height, dtype=torch.float32).view(-1, 1)
        w_coor = torch.linspace(0, width - 1, width, dtype=torch.float32).view(1, -1)
        h_coor = h_coor.repeat(1, width)
        w_coor = w_coor.repeat(height, 1)
        h_coor = h_coor.cuda()
        w_coor = w_coor.cuda()

        for scene_id in tqdm(sel_scene_ids, postfix=f"{split}_{scene}"):
            # print("split: {} scene: {}".format(split, scene_id))
            scene_root = osp.join(data_root, f"{scene_id:06d}")

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            # gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                K = np.array(cam_dict[str_im_id]["cam_K"], dtype="float32").reshape(3, 3)
                camK = K
                camK_inv = np.linalg.inv(camK)
                camK = (torch.from_numpy(K)).cuda()
                camK_inv = (torch.from_numpy(camK_inv)).cuda()
                # from here use space to shorten time

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in idx2class:
                        continue
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = (np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0).reshape(3, 1)
                    R_cuda = torch.from_numpy(R).cuda()
                    t_cuda = torch.from_numpy(t).cuda()
                    RnxT = (torch.matmul(R_cuda, n_x)).permute(1, 0).view(3)  # 1, 3
                    RnyT = (torch.matmul(R_cuda, n_y)).permute(1, 0).view(3)
                    RnzT = (torch.matmul(R_cuda, n_z)).permute(1, 0).view(3)
                    RnxTt = torch.matmul((torch.matmul(R_cuda, n_x).permute(1, 0)), t_cuda).view(1)  # 1,1
                    RnyTt = torch.matmul((torch.matmul(R_cuda, n_y).permute(1, 0)), t_cuda).view(1)
                    RnzTt = torch.matmul((torch.matmul(R_cuda, n_z).permute(1, 0)), t_cuda).view(1)
                    # mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    # assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    # load mask visib  TODO: load both mask_visib and mask_full
                    mask = mmcv.imread(mask_visib_file, "unchanged")
                    mask = mask.astype(np.bool).astype(np.float)
                    mask_cuda = torch.from_numpy(mask).cuda()
                    if mask.sum() < 1:
                        Q0 = {
                            "occ_crop": 0,
                            "xyxy": [0, 0, IM_W - 1, IM_H - 1],
                        }  # in order to save path, not save

                    else:
                        xyz_path = osp.join(self.xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                        assert osp.exists(xyz_path), xyz_path
                        xyz = mmcv.load(xyz_path)
                        x1, y1, x2, y2 = xyz["xyxy"]  # bounding box
                        id = str(obj_id)
                        xmin, xmax, ymin, ymax, zmin, zmax = REC_LIST[id]["xmin"], REC_LIST[id]["xmax"], \
                                                             REC_LIST[id]["ymin"], REC_LIST[id]["ymax"], \
                                                             REC_LIST[id]["zmin"], REC_LIST[id]["zmax"]
                        pd_1 = camK_inv[0, 0] * w_coor + camK_inv[0, 1] * h_coor + camK_inv[0, 2]
                        pd_2 = camK_inv[1, 0] * w_coor + camK_inv[1, 1] * h_coor + camK_inv[1, 2]
                        pd_3 = camK_inv[2, 0] * w_coor + camK_inv[2, 1] * h_coor + camK_inv[2, 2]
                        Z_x_up = RnxTt
                        Z_x_down = RnxT[0] * pd_1 + RnxT[1] * pd_2 + RnxT[2] * pd_3
                        Z_y_up = RnyTt
                        Z_y_down = RnyT[0] * pd_1 + RnyT[1] * pd_2 + RnyT[2] * pd_3
                        Z_z_up = RnzTt
                        Z_z_down = RnzT[0] * pd_1 + RnzT[1] * pd_2 + RnzT[2] * pd_3
                        Z_x_down[torch.abs(Z_x_down) < 1e-6] = 1e-6
                        Z_y_down[torch.abs(Z_y_down) < 1e-6] = 1e-6
                        Z_z_down[torch.abs(Z_z_down) < 1e-6] = 1e-6
                        Q_x_0 = (Z_x_up / Z_x_down) * pd_1  # h, w
                        Q_x_1 = (Z_x_up / Z_x_down) * pd_2
                        Q_x_2 = (Z_x_up / Z_x_down) * pd_3
                        Q_y_0 = (Z_y_up / Z_y_down) * pd_1
                        Q_y_1 = (Z_y_up / Z_y_down) * pd_2
                        Q_y_2 = (Z_y_up / Z_y_down) * pd_3
                        Q_z_0 = (Z_z_up / Z_z_down) * pd_1
                        Q_z_1 = (Z_z_up / Z_z_down) * pd_2
                        Q_z_2 = (Z_z_up / Z_z_down) * pd_3
                        # bounding box cut
                        occ_mask_x, Q0_x_0, Q0_x_1, Q0_x_2 = test_in_box_cuda(Q_x_0, Q_x_1, Q_x_2, xmin, xmax, ymin,
                                                                              ymax,
                                                                              zmin, zmax, R_cuda, t_cuda)
                        occ_mask_y, Q0_y_0, Q0_y_1, Q0_y_2 = test_in_box_cuda(Q_y_0, Q_y_1, Q_y_2, xmin, xmax, ymin,
                                                                              ymax,
                                                                              zmin, zmax, R_cuda, t_cuda)
                        occ_mask_z, Q0_z_0, Q0_z_1, Q0_z_2 = test_in_box_cuda(Q_z_0, Q_z_1, Q_z_2, xmin, xmax, ymin,
                                                                              ymax,
                                                                              zmin, zmax, R_cuda, t_cuda)
                        # mask visib and occ_mask together ->0
                        Q0_x_0[mask_cuda < 1] = 0
                        Q0_x_0[occ_mask_x < 1] = 0
                        Q0_x_1[mask_cuda < 1] = 0
                        Q0_x_1[occ_mask_x < 1] = 0
                        Q0_x_2[mask_cuda < 1] = 0
                        Q0_x_2[occ_mask_x < 1] = 0

                        Q0_y_0[mask_cuda < 1] = 0
                        Q0_y_0[occ_mask_y < 1] = 0
                        Q0_y_1[mask_cuda < 1] = 0
                        Q0_y_1[occ_mask_y < 1] = 0
                        Q0_y_2[mask_cuda < 1] = 0
                        Q0_y_2[occ_mask_y < 1] = 0

                        Q0_z_0[mask_cuda < 1] = 0
                        Q0_z_0[occ_mask_z < 1] = 0
                        Q0_z_1[mask_cuda < 1] = 0
                        Q0_z_1[occ_mask_z < 1] = 0
                        Q0_z_2[mask_cuda < 1] = 0
                        Q0_z_2[occ_mask_z < 1] = 0

                        Q0 = torch.stack((Q0_x_1, Q0_x_2, Q0_y_0, Q0_y_2, Q0_z_0,
                                          Q0_z_1), dim=2)  # H, w, c
                        Q0_output = Q0.cpu().numpy()
                        Q0 = {
                            "occ_crop": Q0_output[y1:y2 + 1, x1:x2 + 1, :],
                            "xyxy": [x1, y1, x2, y2],
                        }
                    Q0_path = osp.join(self.occ_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-Q0.pkl")
                    mmcv.mkdir_or_exist(osp.dirname(Q0_path))
                    mmcv.dump(Q0, Q0_path)



def run_real_one():
    import argparse
    import time

    import setproctitle
    import torch

    parser = argparse.ArgumentParser(description="gen ycbv bop xyz")
    parser.add_argument("--split", type=str, default="train_real", help="split")
    parser.add_argument("--scene", type=str, default="all", help="scene id")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--vis", default=False, action="store_true", help="vis")
    args = parser.parse_args()

    height = IM_H
    width = IM_W

    VIS = args.vis

    device = torch.device(int(args.gpu))
    dtype = torch.float32
    tensor_kwargs = {"device": device, "dtype": dtype}

    T_begin = time.perf_counter()
    setproctitle.setproctitle(f"gen_xyz_ycbv_{args.split}_{args.scene}")
    xyz_gen = XyzGen(args.split, args.scene)
    xyz_gen.main()
    T_end = time.perf_counter() - T_begin
    print("split", args.split, "scene", args.scene, "total time: ", get_time_delta(T_end))

if __name__ == "__main__":
    import argparse
    import time

    import setproctitle
    import torch

    parser = argparse.ArgumentParser(description="gen ycbv bop xyz")
    parser.add_argument("--split", type=str, default="train_real", help="split")
    parser.add_argument("--scene", type=str, default="all", help="scene id")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--vis", default=False, action="store_true", help="vis")
    args = parser.parse_args()

    height = IM_H
    width = IM_W

    VIS = args.vis

    device = torch.device(int(args.gpu))
    dtype = torch.float32
    tensor_kwargs = {"device": device, "dtype": dtype}

    T_begin = time.perf_counter()
    setproctitle.setproctitle(f"gen_xyz_ycbv_{args.split}_{args.scene}")
    xyz_gen = XyzGen(args.split, args.scene)
    xyz_gen.main()
    T_end = time.perf_counter() - T_begin
    print("split", args.split, "scene", args.scene, "total time: ", get_time_delta(T_end))
