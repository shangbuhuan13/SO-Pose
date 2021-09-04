import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from datetime import timedelta

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
a = 49
scenes = [i for i in range(a, a+1)]
xyz_root = "/data/wanggu/Storage/BOP_DATASETS/ycbv/train_pbr/xyz_crop"

K = np.array([[1066.778, 0.0, 312.9869079589844], [0.0, 1067.487, 241.3108977675438], [0.0, 0.0, 1.0]], dtype=np.float32)

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
        self.renderer = None

    def get_renderer(self):
        if self.renderer is None:
            self.renderer = EGLRenderer(
                model_paths,
                texture_paths=texture_paths,
                vertex_scale=0.001,
                height=IM_H,
                width=IM_W,
                znear=near,
                zfar=far,
                use_cache=True,
                gpu_id=int(args.gpu),
            )
            self.image_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
            self.seg_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
            # self.pc_obj_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
            self.depth_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
        return self.renderer

    def main(self):
        split = self.split
        scene = self.scene  # "all" or a single scene
        sel_scene_ids = self.sel_scene_ids
        data_root = self.data_root
        camK = K
        camK_inv = np.linalg.inv(camK)
        camK = (torch.from_numpy(K)).cuda()
        camK_inv = (torch.from_numpy(camK_inv)).cuda()
        height = 480
        width = 640
        h_coor = torch.linspace(0, height - 1, height, dtype=torch.float32).view(-1, 1)
        w_coor = torch.linspace(0, width - 1, width, dtype=torch.float32).view(1, -1)
        h_coor = h_coor.repeat(1, width)
        w_coor = w_coor.repeat(height, 1)
        h_coor = h_coor.cuda()
        w_coor = w_coor.cuda()
        for scene_id in tqdm(sel_scene_ids, postfix=f"{split}_{scene}"):
            print("split: {} scene: {}".format(split, scene_id))
            scene_root = osp.join(data_root, f"{scene_id:06d}")

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            #gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            #cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in idx2class:
                        continue

                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                    t = t.reshape(3, 1)
                    pose = np.hstack([R, t])

                    save_path = osp.join(xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                    # if osp.exists(save_path) and osp.getsize(save_path) > 0:
                    #     continue

                    render_obj_id = cls_indexes.index(obj_id)  # 0-based
                    self.get_renderer().render(
                        [render_obj_id],
                        [pose],
                        K=K,
                        image_tensor=self.image_tensor,
                        seg_tensor=self.seg_tensor,
                        pc_obj_tensor=None,
                        pc_cam_tensor=self.depth_tensor
                    )

                    if VIS:
                        bgr_gl = (self.image_tensor[:, :, :3].cpu().numpy() + 0.5).astype(np.uint8)

                    mask = (self.seg_tensor[:, :, 0] > 0).to(torch.uint8)
                    if mask.sum() == 0:  # NOTE: this should be ignored at training phase
                        '''
                        print(
                            f"not visible, split {split} scene {scene_id}, im {int_im_id} obj {idx2class[obj_id]} {obj_id}"
                        )
                        '''
                        P = {
                            "xyz_crop": np.zeros((IM_H, IM_W, 3), dtype=np.float16),
                            "xyxy": [0, 0, IM_W - 1, IM_H - 1],
                        }
                        if VIS:
                            im_path = osp.join(data_root, f"{scene_id:06d}/rgb/{int_im_id:06d}.jpg")
                            im = mmcv.imread(im_path)

                            mask_path = osp.join(data_root, f"{scene_id:06d}/mask/{int_im_id:06d}_{anno_i:06d}.png")
                            mask_visib_path = osp.join(
                                data_root, f"{scene_id:06d}/mask_visib/{int_im_id:06d}_{anno_i:06d}.png"
                            )
                            mask_gt = mmcv.imread(mask_path, "unchanged")
                            mask_visib_gt = mmcv.imread(mask_visib_path, "unchanged")

                            #show_ims = [bgr_gl[:, :, [2, 1, 0]], im[:, :, [2, 1, 0]], mask_gt, mask_visib_gt]
                            #show_titles = ["bgr_gl", "im", "mask_gt", "mask_visib_gt"]
                            #grid_show(show_ims, show_titles, row=2, col=2)
                            #raise RuntimeError(f"split {split} scene {scene_id}, im {int_im_id}")
                    else:
                        ys_xs = mask.nonzero(as_tuple=False)
                        ys, xs = ys_xs[:, 0], ys_xs[:, 1]
                        x1, y1 = [xs.min().item(), ys.min().item()]
                        x2, y2 = [xs.max().item(), ys.max().item()]

                        depth_th = self.depth_tensor[:, :, 2].detach()  # h, w tensor
                        pd_1 = camK_inv[0, 0] * w_coor + camK_inv[0, 1] * h_coor + camK_inv[0, 2]
                        pd_2 = camK_inv[1, 0] * w_coor + camK_inv[1, 1] * h_coor + camK_inv[1, 2]
                        pd_3 = camK_inv[2, 0] * w_coor + camK_inv[2, 1] * h_coor + camK_inv[2, 2]
                        P_1 = pd_1 * depth_th
                        P_2 = pd_2 * depth_th
                        P_3 = pd_3 * depth_th
                        R_cuda = torch.from_numpy(R).cuda()
                        t_cuda = torch.from_numpy(t).cuda()
                        RT_cuda = R_cuda.permute(1, 0)
                        RTt_cuda = torch.matmul(RT_cuda, t_cuda).view(3)  # 3, 1
                        P0_1 = RT_cuda[0, 0] * P_1 + RT_cuda[0, 1] * P_2 + RT_cuda[0, 2] * P_3 - RTt_cuda[0]
                        P0_2 = RT_cuda[1, 0] * P_1 + RT_cuda[1, 1] * P_2 + RT_cuda[1, 2] * P_3 - RTt_cuda[1]
                        P0_3 = RT_cuda[2, 0] * P_1 + RT_cuda[2, 1] * P_2 + RT_cuda[2, 2] * P_3 - RTt_cuda[2]
                        P0 = torch.stack((P0_1, P0_2, P0_3), dim=2)
                        # from depth_th to new xyz_th
                        #depth_th = depth_th.cpu().numpy()
                        '''
                        P0_output = np.zeros([height, width, 3], dtype=np.float32)
                        for i in range(y1, y2 + 1):
                            for j in range(x1, x2 + 1):
                                if mask[i][j] < 1:
                                    continue
                                else:
                                    point = np.array([j, i, 1])
                                    point = point.reshape(3, 1)  # 3,1
                                  
                                    ids = np.arange(0, vert_id.shape[0])
                                    selected = ids[flag]
                                    s_a = A[selected, :]
                                    s_b =B[selected, :]
                                    s_c = C[selected,:]
                                    flag = pointintriangle(s_a, s_b, s_c, point[0:2, :])
                                  
                                    point_diretion = np.matmul(camK_inv, point)  # 3, 1
                                    Z_p_final = depth_th[i][j]
                                    P = (Z_p_final * point_diretion)
                                    P0 = transformer_back(P, R, t)
                                    # P0_3 = P0.reshape(3)
                                    P0_output[i, j, :] = P0.reshape(3)
                                    ## r, t transform
                        '''
                        P0[mask < 1, :] = 0.0
                        P0_output = P0.cpu().numpy()
                        xyz_crop = P0_output[y1 : y2 + 1, x1 : x2 + 1]
                        P = {"xyz_crop": xyz_crop.astype("float16"), "xyxy": [x1, y1, x2, y2]}

                    mmcv.mkdir_or_exist(osp.dirname(save_path))
                    mmcv.dump(P, save_path)
        if self.renderer is not None:
            self.renderer.close()


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
