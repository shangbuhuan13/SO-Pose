import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from datetime import timedelta
from plyfile import PlyData
import mmcv
import numpy as np
from tqdm import tqdm

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

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/ycbv/train_pbr"))

cls_indexes = sorted(idx2class.keys())
cls_names = [idx2class[cls_idx] for cls_idx in cls_indexes]
model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/ycbv/models"))
model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in cls_indexes]
texture_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.png") for obj_id in cls_indexes]

scenes = [i for i in range(0, 49 + 1)]
save_root = "/data/wanggu/Storage"
xyz_root = osp.normpath(osp.join(save_root, "BOP_DATASETS/ycbv/train_pbr/xyz_crop"))

K = np.array([[1066.778, 0.0, 312.9869079589844], [0.0, 1067.487, 241.3108977675438], [0.0, 0.0, 1.0]], dtype=np.float32)

def transformer(P0, R, t):  # P0, n,3 torch.tensor
    P = (torch.mm(R, P0.transpose(1, 0))).T + t.view(1, 3)
    return P


def transformer_back(P, R, t):  # 计算P0=RTP-RTt  P, 3*1
    P0 = torch.mm(R.transpose(1, 0), P) - torch.mm(R.transpose(1, 0), t)
    return P0

# P0 n,3
def projector(P0, K, R, t):  # 计算相机投影， 将P0经过R， t变换再投影到图像上, torch.tensor
    p = (torch.mm(K, P0.transpose(1, 0))).transpose(1, 0) / P0[:, 2:]  # n,3
    p = p[:, 0:2] / p[:, 2:]
    return p


def pointintriangle(A, B, C, P):  # 判断一点是否在3角面片内部，ABC是三角面片3个点
    # P :2*1  A,B,C : n,2
    v0 = C - A  # n,2
    v1 = B - A
    v2 = P.transpose(1, 0) - A

    dot00 = torch.sum(v0 * v0, dim=1)  # n,1
    dot01 = torch.sum(v0 * v1, dim=1)
    dot02 = torch.sum(v0 * v2, dim=1)
    dot11 = torch.sum(v1 * v1, dim=1)
    dot12 = torch.sum(v1 * v2, dim=1)

    down = dot00 * dot11 - dot01 * dot01  # n,1
    down_save = down.clone()
    down[down < 1e-7] = 1
    inverdeno = 1 / down
    u = (dot11 * dot02 - dot01 * dot12) * inverdeno  # n,1
    v = (dot00 * dot12 - dot01 * dot02) * inverdeno
    mask = (down_save > 1e-6) & (u >= 0) & (v >= 0) & (u+v <= 1)
    return mask


def modelload(model_dir, ids, scale=1000.):
    modellist = {}
    for obj in ids:
        print("loading model", f"{obj:06d}")
        model_path = osp.join(model_dir, "obj_{:06d}.ply".format(obj))
        ply = PlyData.read(model_path)
        vert = np.asarray(
            [ply['vertex'].data['x'] / scale, ply['vertex'].data['y'] / scale,
             ply['vertex'].data['z'] / scale]).transpose()
        norm_d = np.asarray(
            [ply['vertex'].data['nx'], ply['vertex'].data['ny'], ply['vertex'].data['nz']]).transpose()
        vert_id = [id for id in ply['face'].data['vertex_indices']]
        vert_id = np.asarray(vert_id, np.int64)
        modellist[str(obj)] = {
            "vert": torch.as_tensor(vert.astype("float32")).cuda(),
            "norm_d": torch.as_tensor(norm_d.astype("float32")).cuda(),
            "vert_id": torch.as_tensor(vert_id.astype("int64")).cuda(),
        }
    return modellist

def get_time_delta(sec):
    """Humanize timedelta given in seconds."""
    if sec < 0:
        return "{:.3g} seconds".format(sec)
    delta_time_str = str(timedelta(seconds=sec))
    return delta_time_str

class XyzGen(object):
    def __init__(self, split="train", scene="all"):
        if split == "train":
            scene_ids = scenes
            data_root = data_dir
        else:
            raise ValueError(f"split {split} error")

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
        self.model = modelload(model_dir, cls_indexes)
        self.xyz_root = "/data/wanggu/Storage/BOP_DATASETS/ycbv/train_pbr/xyz_crop"

    def main(self):
        split = self.split
        scene = self.scene  # "all" or a single scene
        sel_scene_ids = self.sel_scene_ids
        data_root = self.data_root
        camK_inv = np.linalg.inv(K)
        camK_cuda = torch.as_tensor(K).cuda()
        camK_inv_cuda = torch.as_tensor(camK_inv).cuda()
        height = 480
        width = 640
        for scene_id in tqdm(sel_scene_ids, postfix=f"{split}_{scene}"):
            print("split: {} scene: {}".format(split, scene_id))
            scene_root = osp.join(data_root, f"{scene_id:06d}")

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in idx2class:
                        continue
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = (np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0).reshape(3, 1)
                    R_cuda = torch.as_tensor(R).cuda()
                    t_cuda = torch.as_tensor(t).cuda()
                    # mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    # assert osp.exists(mask_file), mask_file
                    # load mask visib  TODO: load both mask_visib and mask_full
                    mask = mmcv.imread(mask_visib_file, "unchanged")
                    mask = mask.astype(np.bool).astype(np.float)
                    mask_cuda = torch.as_tensor(mask).cuda()
                    if torch.sum(mask_cuda) == 0:
                        P = {
                            "xyz_crop": np.zeros((height, width, 3), dtype=np.float16),
                            "xyxy": [0, 0, width - 1, height - 1],
                        }

                    else:

                        xyz_path = osp.join(self.xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")

                        assert osp.exists(xyz_path), xyz_path
                        xyz = mmcv.load(xyz_path)
                        # begin to estimate new xyz
                        # 实际上就是将每个3角面片投影回来，查看其中包含的整点像素，为其提供一个估计，然后最后选择能看到的那个，Z值最小
                        vert = torch.as_tensor(self.model[str(obj_id)]["vert"])
                        norm_d = torch.as_tensor(self.model[str(obj_id)]["norm_d"])
                        vert_id = torch.as_tensor(self.model[str(obj_id)]["vert_id"])
                        # project all vertices
                        vert_trans = transformer(vert, R_cuda, t_cuda)  # already on cuda
                        d_trans = (torch.mm(R_cuda, (norm_d[vert_id[:, 0], :]).transpose(1, 0))).transpose(1, 0)
                        vert_proj = projector(vert_trans, camK_cuda, R_cuda, t_cuda)  # n, 2
                        # construct triangle
                        A_before = vert_trans[vert_id[:, 0], :]  # already on cuda
                        A = vert_proj[vert_id[:, 0], :]
                        B = vert_proj[vert_id[:, 1], :]
                        C = vert_proj[vert_id[:, 2], :]

                        # 生成P0的图， 之前只存储了Zp， 现在计算值
                        # pixellist is the result
                        P0_output = torch.zeros([height, width, 3], dtype=torch.float32).cuda()
                        # np.zeros([height, width, 3], dtype=np.float32)
                        x1, y1, x2, y2 = xyz["xyxy"]
                        for i in range(y1, y2 + 1):
                            for j in range(x1, x2 + 1):
                                if mask_cuda[i][j] < 1:
                                    continue
                                else:
                                    point = torch.as_tensor([j, i, 1], dtype=torch.float32).cuda()
                                    point = point.view(3, 1)  # 3,1
                                    flag = pointintriangle(A, B, C, point[0:2, :])
                                    if torch.sum(flag) == 0:
                                        continue
                                    '''for test'''
                                    '''
                                    ids = np.arange(0, vert_id.shape[0])
                                    selected = ids[flag]
                                    s_a = A[selected, :]
                                    s_b =B[selected, :]
                                    s_c = C[selected,:]
                                    flag = pointintriangle(s_a, s_b, s_c, point[0:2, :])
                                    '''
                                    ''''''
                                    p_A = A_before[flag, :]
                                    p_d = d_trans[flag, :]
                                    plane_d = torch.sum(p_d * p_A, dim=1).view(-1, 1)  # m, 1
                                    point_diretion = torch.mm(camK_inv_cuda, point)  # 3, 1
                                    down_Z = torch.mm(p_d, point_diretion)  # m, 1
                                    Z_p = plane_d / down_Z
                                    Z_p = torch.abs(Z_p)
                                    Z_p_final = torch.min(Z_p)
                                    P = (Z_p_final * point_diretion)
                                    P0 = transformer_back(P, R_cuda, t_cuda)
                                    # P0_3 = P0.reshape(3)
                                    P0_output[i, j, :] = P0.reshape(3)  # 边界上的点在计算的时候会出现错误， 没有完全包裹住

                        # xyz_value = xyz["xyz_crop"]
                        P0_res = P0_output.cpu().numpy()
                        P = {
                            "xyz_crop": P0_output[y1:y2 + 1, x1:x2 + 1, :].cpu().numpy(),
                            "xyxy": [x1, y1, x2, y2],
                        }
                        xyz = 1


if __name__ == "__main__":
    import argparse
    import time

    import setproctitle
    import torch

    parser = argparse.ArgumentParser(description="gen ycbv train_pbr xyz")
    parser.add_argument("--split", type=str, default="train", help="split")
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
    setproctitle.setproctitle(f"gen_xyz_ycbv_train_pbr_{args.split}_{args.scene}")
    xyz_gen = XyzGen(args.split, args.scene)
    xyz_gen.main()
    T_end = time.perf_counter() - T_begin
    print("split", args.split, "scene", args.scene, "total time: ", get_time_delta(T_end))
