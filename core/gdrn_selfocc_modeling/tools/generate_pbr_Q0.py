# 产生Q0坐标，同时计算occlusion mask
import sys
import os.path as osp
sys.path.append('../')
import numpy as np
from PIL import Image, ImageFile
import os
import matplotlib.image as mp
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mmcv
import ref
from tqdm import tqdm
from collections import OrderedDict

# 需要知道物体的外接矩形信息，在那个models_info.json里面
def read_rec(model_info_path, obj_id):
    id = obj_id
    model_info = mmcv.load(model_info_path)
    diameter = model_info[id]["diameter"]
    x_min, x_size = model_info[id]["min_x"], model_info[id]["size_x"]
    y_min, y_size = model_info[id]["min_y"], model_info[id]["size_y"]
    z_min, z_size = model_info[id]["min_z"], model_info[id]["size_z"]
    return diameter, x_min, x_size, y_min, y_size, z_min, z_size

def test_in_box(point, xmin, xmax, ymin, ymax, zmin, zmax, R, t):
    # 要先将点坐标变换回去
    point = np.matmul(R.T, point) - np.matmul(R.T, t)
    # print(point)
    if xmin < point[0] < xmax and ymin < point[1] < ymax and zmin < point[2] < zmax:
        return 1, point
    else:
        return 0, 0

LM_13_OBJECTS = [
    "ape",
    "benchvise",
    "camera",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
    "iron",
    "lamp",
    "phone",
]  # no bowl, cup
LM_OCC_OBJECTS = ["ape", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]

class Q0_generator():
    def __init__(self, rootdir, modeldir): # /data/wanggu//Storage/BOP_DATASETS/lm/train_pbr
        self.dataset_root = rootdir
        self.modeldir = modeldir
        # NOTE: careful! Only the selected objects
        self.objs = LM_OCC_OBJECTS
        self.cat_ids = [cat_id for cat_id, obj_name in ref.lm_full.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        self.scenes = [f"{i:06d}" for i in range(50)]
        self.xyz_root = osp.join(self.dataset_root, "xyz_crop")
    def run(self, scale=1000):
        for scene in self.scenes:
            scene_id = int(scene)
            scene_root = osp.join(self.dataset_root, scene)

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            Q0_path = osp.join(self.dataset_root, "Q0", scene)
            if not os.path.exists(Q0_path):
                os.makedirs(Q0_path)
            for str_im_id in gt_dict.keys():
                int_im_id = int(str_im_id)
                print("processing seq:{:06d} obj:{:06d}".format(scene_id, int_im_id))
                rgb_path = osp.join(scene_root, "rgb/{:06d}.jpg").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path

                '''
                show image
                rgb = mmcv.imread(rgb_path, "unchanged")
                plt.imshow(rgb)
                plt.show()
                '''

                camK = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
                camK_inv = np.linalg.inv(camK)
                depth_factor = 1000.0 / cam_dict[str_im_id]["depth_scale"]  # 10000

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in self.cat_ids:
                        continue
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = (np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0).reshape(3, 1)

                    # mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    # assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    # load mask visib  TODO: load both mask_visib and mask_full
                    mask = mmcv.imread(mask_visib_file, "unchanged")
                    mask = mask.astype(np.bool).astype(np.float)
                    area = mask.sum()
                    '''
                    show mask
                    plt.imshow(mask)
                    plt.show()
                    '''
                    xyz_path = osp.join(self.xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                    assert osp.exists(xyz_path), xyz_path
                    xyz = mmcv.load(xyz_path)
                    x1, y1, x2, y2 = xyz["xyxy"]
                    model_info_path = osp.join(self.modeldir, "models_info.json")
                    diameter, xmin, x_size, ymin, y_size, zmin, z_size = read_rec(model_info_path, str(12))
                    xmax = xmin + x_size
                    ymax = ymin + y_size
                    zmax = zmin + z_size
                    xmin = xmin / scale
                    xmax = xmax / scale
                    ymin = ymin / scale
                    ymax = ymax / scale
                    zmin = zmin / scale
                    zmax = zmax / scale
                    # 开始循环
                    height, width = mask.shape
                    #  存储遮挡mask
                    occ_mask_x = np.zeros((height, width))
                    occ_mask_y = np.zeros((height, width))
                    occ_mask_z = np.zeros((height, width))
                    # 存储Q0的坐标
                    Q0_x = np.zeros((3, height, width))
                    Q0_y = np.zeros((3, height, width))
                    Q0_z = np.zeros((3, height, width))
                    n_x = np.array([[1], [0], [0]])  # Q0_yz
                    n_y = np.array([[0], [1], [0]])  # Q0_xz
                    n_z = np.array([[0], [0], [1]])  # Q0_xy
                    # 计算一些必要的量
                    RnxTt = np.matmul(np.matmul(R, n_x).T, t)
                    RnyTt = np.matmul(np.matmul(R, n_y).T, t)
                    RnzTt = np.matmul(np.matmul(R, n_z).T, t)
                    for i in range(height):
                        for j in range(width):
                            point = np.array([[j], [i], [1]])
                            if mask[i][j] < 1:
                                continue
                            else:
                                Q0_x_v = (RnxTt / np.matmul(np.matmul(R, n_x).T,
                                                            np.matmul(camK_inv, point))) * np.matmul(camK_inv, point)
                                occ_mask_x[i][j], Q_save = test_in_box(Q0_x_v, xmin, xmax, ymin, ymax, zmin, zmax, R, t)
                                if occ_mask_x[i][j] > 0:
                                    Q0_x[:, i, j] = Q_save.squeeze()
                                Q0_y_v = (RnyTt / np.matmul(np.matmul(R, n_y).T,
                                                            np.matmul(camK_inv, point))) * np.matmul(camK_inv, point)
                                occ_mask_y[i][j], Q_save = test_in_box(Q0_y_v, xmin, xmax, ymin, ymax, zmin, zmax, R, t)
                                if occ_mask_y[i][j] > 0:
                                    Q0_y[:, i, j] = Q_save.squeeze()
                                Q0_z_v = (RnzTt / np.matmul(np.matmul(R, n_z).T,
                                                            np.matmul(camK_inv, point))) * np.matmul(camK_inv, point)
                                occ_mask_z[i][j], Q_save = test_in_box(Q0_z_v, xmin, xmax, ymin, ymax, zmin, zmax, R, t)
                                if occ_mask_z[i][j] > 0:
                                    Q0_z[:, i, j] = Q_save.squeeze()

                    Q0 = np.concatenate((Q0_x[1:, :, :], Q0_y[0:1, :, :], Q0_y[2:, :, :], Q0_z[:2, :, :]), axis=0)
                    # 维度变一下CHW -  HWC
                    Q0 = Q0.transpose((1, 2, 0))
                    Q0 = {
                        "occ_crop": Q0[y1:y2 + 1, x1:x2 + 1, :],
                        "xyxy": [x1, y1, x2, y2],
                    }
                    #  存储 Q0的坐标
                    outpath = os.path.join(Q0_path, "{:06d}_{:06d}-Q0.pkl".format(int_im_id, anno_i))
                    mmcv.dump(Q0, outpath)


if __name__ == "__main__":
    root_dir = "/data/wanggu/Storage/BOP_DATASETS/lm/train_pbr"
    model_dir = "/data/wanggu/Storage/BOP_DATASETS/lm/models"
    G_Q = Q0_generator(root_dir, model_dir)
    G_Q.run(scale=1000)