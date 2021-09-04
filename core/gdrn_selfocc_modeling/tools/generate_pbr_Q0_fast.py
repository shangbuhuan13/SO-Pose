
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

# in models_info.json
def read_rec(model_info_path, obj_id):
    id = obj_id
    model_info = mmcv.load(model_info_path)
    diameter = model_info[id]["diameter"]
    x_min, x_size = model_info[id]["min_x"], model_info[id]["size_x"]
    y_min, y_size = model_info[id]["min_y"], model_info[id]["size_y"]
    z_min, z_size = model_info[id]["min_z"], model_info[id]["size_z"]
    return diameter, x_min, x_size, y_min, y_size, z_min, z_size

def test_in_box(point, xmin, xmax, ymin, ymax, zmin, zmax, R, t):

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
REC_LIST = {
    '1':{"xmin":-0.0379343, "xmax":0.0379343, "ymin":-0.0387996, "ymax":0.0387996, "zmin":-0.0458845, "zmax":0.0458845},
    '5':{"xmin":-0.0503958, "xmax":0.0503962, "ymin":-0.0908979, "ymax":0.0908981, "zmin":-0.0968670, "zmax":0.0968670},
    '6':{"xmin":-0.0335054, "xmax":0.0335053, "ymin":-0.0638165, "ymax":0.0638165, "zmin":-0.0587283, "zmax":0.0587283},
    '8':{"xmin":-0.1147380, "xmax":0.1147380, "ymin":-0.0377357, "ymax":0.0377357, "zmin":-0.1040010, "zmax":0.1040010},
    '9':{"xmin":-0.0522146, "xmax":0.0522146, "ymin":-0.0387038, "ymax":0.0387038, "zmin":-0.0428485, "zmax":0.0428485},
    '10':{"xmin":-0.0750923, "xmax":0.0750927, "ymin":-0.0535375, "ymax":0.0535375, "zmin":-0.0346207, "zmax":0.0346207},
    '11':{"xmin":-0.0183605, "xmax":0.0183606, "ymin":-0.0389330, "ymax":0.0389330, "zmin":-0.0864079, "zmax":0.0864081},
    '12':{"xmin":-0.0504439, "xmax":0.0504441, "ymin":-0.0542485, "ymax":0.0542485, "zmin":-0.0454000, "zmax":0.0454000},
}

intrinsic_matrix = {
    'linemod': np.array([[572.4114, 0., 325.2611],
                         [0., 573.57043, 242.04899],
                         [0., 0., 1.]]),
}

class Q0_generator_fast():
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
        self.scenes = [f"{i:06d}" for i in range(24, 50)]
        self.xyz_root = osp.join(self.dataset_root, "xyz_crop")
    def run(self, scale=1000):
        camK = intrinsic_matrix["linemod"].astype(np.float32)
        camK_inv = np.linalg.inv(camK)
        height = 480
        width = 640
        for scene in self.scenes:
            scene_id = int(scene)
            scene_root = osp.join(self.dataset_root, scene)
            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            #gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            # cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))
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
                # from here use space to shorten time
                obj_id_save = []
                mask_save = []
                R_save = []
                t_save = []
                xyz_save = []
                flag_save = []
                mask_all = np.zeros((height, width))
                n_x = np.array([[1.], [0.], [0.]])  # Q0_yz
                n_y = np.array([[0.], [1.], [0.]])  # Q0_xz
                n_z = np.array([[0.], [0.], [1.]])  # Q0_xy
                RnxTt_save = []
                RnyTt_save = []
                RnzTt_save = []

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id in self.cat_ids:
                        R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                        t = (np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0).reshape(3, 1)
                        RnxTt = np.matmul(np.matmul(R, n_x).T, t)
                        RnyTt = np.matmul(np.matmul(R, n_y).T, t)
                        RnzTt = np.matmul(np.matmul(R, n_z).T, t)
                        # mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                        mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                        # assert osp.exists(mask_file), mask_file
                        assert osp.exists(mask_visib_file), mask_visib_file
                        # load mask visib  TODO: load both mask_visib and mask_full
                        mask = mmcv.imread(mask_visib_file, "unchanged")
                        mask = mask.astype(np.bool).astype(np.float)
                        mask_all = mask_all + mask
                        '''
                        show mask
                        plt.imshow(mask)
                        plt.show()
                        '''
                        xyz_path = osp.join(self.xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                        assert osp.exists(xyz_path), xyz_path
                        xyz = mmcv.load(xyz_path)
                        mask_save.append(mask)
                        R_save.append(R)
                        t_save.append(t)
                        xyz_save.append(xyz)
                        flag_save.append(anno_i)
                        obj_id_save.append(obj_id)
                        RnxTt_save.append(RnxTt)
                        RnyTt_save.append(RnyTt)
                        RnzTt_save.append(RnzTt)


                # generate Qo in a single iteration
                numK = len(flag_save)
                mask_all = mask_all.astype(np.bool).astype(np.float)
                Q0_x = np.zeros((numK*3, height, width))
                Q0_y = np.zeros((numK*3, height, width))
                Q0_z = np.zeros((numK*3, height, width))

                for i in range(height):
                    for j in range(width):
                        point = np.array([[j], [i], [1]])
                        if mask_all[i][j] < 1:
                            continue
                        for g in range(numK):
                            mask = mask_save[g]
                            if mask[i][j] < 1:
                                continue
                            R = R_save[g]
                            t = t_save[g]
                            obj_id = obj_id_save[g]
                            RnxTt = RnxTt_save[g]
                            RnyTt = RnyTt_save[g]
                            RnzTt = RnzTt_save[g]
                            obj_id = str(obj_id)
                            xmin, xmax, ymin, ymax, zmin, zmax = REC_LIST[obj_id]["xmin"], REC_LIST[obj_id]["xmax"],\
                                                                    REC_LIST[obj_id]["ymin"], REC_LIST[obj_id]["ymax"],\
                                                                    REC_LIST[obj_id]["zmin"], REC_LIST[obj_id]["zmax"]
                            Q0_x_v = (RnxTt / np.matmul(np.matmul(R, n_x).T,
                                                        np.matmul(camK_inv, point))) * np.matmul(camK_inv, point)
                            occ_mask_x, Q_save = test_in_box(Q0_x_v, xmin, xmax, ymin, ymax, zmin, zmax, R, t)
                            if occ_mask_x > 0:
                                Q0_x[g*3:(g+1)*3, i, j] = Q_save.squeeze()
                            Q0_y_v = (RnyTt / np.matmul(np.matmul(R, n_y).T,
                                                        np.matmul(camK_inv, point))) * np.matmul(camK_inv, point)
                            occ_mask_y, Q_save = test_in_box(Q0_y_v, xmin, xmax, ymin, ymax, zmin, zmax, R, t)
                            if occ_mask_y > 0:
                                Q0_y[g*3:(g+1)*3, i, j] = Q_save.squeeze()
                            Q0_z_v = (RnzTt / np.matmul(np.matmul(R, n_z).T,
                                                        np.matmul(camK_inv, point))) * np.matmul(camK_inv, point)
                            occ_mask_z, Q_save = test_in_box(Q0_z_v, xmin, xmax, ymin, ymax, zmin, zmax, R, t)
                            if occ_mask_z > 0:
                                Q0_z[g*3:(g+1)*3, i, j] = Q_save.squeeze()
                # save all Q0
                for g in range(numK):
                    xyz = xyz_save[g]
                    x1, y1, x2, y2 = xyz["xyxy"]
                    Q0_x_save = Q0_x[g * 3:(g + 1) * 3, :, :]
                    Q0_y_save = Q0_y[g * 3:(g + 1) * 3, :, :]
                    Q0_z_save = Q0_z[g * 3:(g + 1) * 3, :, :]
                    Q0 = np.concatenate((Q0_x_save[1:, :, :], Q0_y_save[0:1, :, :], Q0_y_save[2:, :, :],
                                         Q0_z_save[:2, :, :]), axis=0)
                    # 维度变一下CHW -  HWC
                    Q0 = Q0.transpose((1, 2, 0))
                    Q0 = {
                        "occ_crop": Q0[y1:y2 + 1, x1:x2 + 1, :],
                        "xyxy": [x1, y1, x2, y2],
                    }
                    #  存储 Q0的坐标
                    anno_i = flag_save[g]
                    outpath = os.path.join(Q0_path, "{:06d}_{:06d}-Q0.pkl".format(int_im_id, anno_i))
                    mmcv.dump(Q0, outpath)


if __name__ == "__main__":
    root_dir = "/data/wanggu/Storage/BOP_DATASETS/lm/train_pbr"
    model_dir = "/data/wanggu/Storage/BOP_DATASETS/lm/models"
    G_Q = Q0_generator_fast(root_dir, model_dir)
    G_Q.run(scale=1000)