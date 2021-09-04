# 产生Q0坐标，同时计算occlusion mask
import sys

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


# 把mask信息读进来
def read_mask_np(mask_pth):  # 读取mask文件, 转换成0-1文件的整数
    mask = Image.open(mask_pth).convert('1')
    mask_seg = np.array(mask).astype(np.int32)
    return mask_seg


# 需要知道物体的外接矩形信息，在那个models_info.json里面
def read_rec(model_info_path, obj_name):
    id = ref.lm_full.obj2id[obj_name]
    id = str(id)
    model_info = mmcv.load(model_info_path)
    diameter = model_info[id]["diameter"]
    x_min, x_size = model_info[id]["min_x"], model_info[id]["size_x"]
    y_min, y_size = model_info[id]["min_y"], model_info[id]["size_y"]
    z_min, z_size = model_info[id]["min_z"], model_info[id]["size_z"]
    return diameter, x_min, x_size, y_min, y_size, z_min, z_size


# 把pose信息整体加载进来，我们这个是一个object一个object来处理的
def read_pose_np(pose_path):  # 读取对应的pose
    pose_info = mmcv.load(pose_path)
    return pose_info


def transformer(P0, R, t):  # 计算P=Rp0+t
    P0 = np.reshape(P0, [3, 1])
    P = np.matmul(R, P0) + t
    return P


def transformer_back(P, R, t):  # 计算P0=RTP-RTt
    P0 = np.matmul(R.T, P) - np.matmul(R.T, t)
    return P0


def projector(P0, K, R, t):  # 计算相机投影， 将P0经过R， t变换再投影到图像上
    p = np.matmul(K, P0) / P0[2]
    p = p[0:2, :] / p[2]
    return p


def pointintriangle(A, B, C, P):  # 判断一点是否在3角面片内部，ABC是三角面片3个点
    P = np.expand_dims(P, 1)
    v0 = C - A
    v1 = B - A
    v2 = P - A

    dot00 = np.matmul(v0.T, v0)
    dot01 = np.matmul(v0.T, v1)
    dot02 = np.matmul(v0.T, v2)
    dot11 = np.matmul(v1.T, v1)
    dot12 = np.matmul(v1.T, v2)

    inverdeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverdeno
    if u < 0 or u > 1:
        return False
    v = (dot00 * dot12 - dot01 * dot02) * inverdeno
    if v < 0 or v > 1:
        return False
    return u + v <= 1


def test_in_box(point, xmin, xmax, ymin, ymax, zmin, zmax, R, t):
    # 要先将点坐标变换回去
    point = np.matmul(R.T, point) - np.matmul(R.T, t)
    # print(point)
    if xmin < point[0] < xmax and ymin < point[1] < ymax and zmin < point[2] < zmax:
        return 1, point
    else:
        return 0, 0


intrinsic_matrix = {
    'linemod': np.array([[572.4114, 0., 325.2611],
                         [0., 573.57043, 242.04899],
                         [0., 0., 1.]]),
    'blender': np.array([[700., 0., 320.],
                         [0., 700., 240.],
                         [0., 0., 1.]]),
    'pascal': np.asarray([[-3000.0, 0.0, 0.0],
                          [0.0, 3000.0, 0.0],
                          [0.0, 0.0, 1.0]])
}


def estimate_occ_mask_Q0(rootdir, cls_name, scale=1000):  # cls_name也就是objectname
    # create basic path if not exist
    id = ref.lm_full.obj2id[cls_name]  # id, 物体的编号
    basic_path = os.path.join(rootdir, "test/Q0/{:06d}".format(id))
    if not os.path.exists(basic_path):
        os.makedirs(basic_path)

    img_dir = os.path.join(rootdir, "test", "{:06d}".format(id), "rgb")
    img_num = len(os.listdir(img_dir))  # 得到图片的数量
    model_info_path = os.path.join(rootdir, "models/models_info.json")
    diameter, xmin, x_size, ymin, y_size, zmin, z_size = read_rec(model_info_path, cls_name)
    xmax = xmin + x_size
    ymax = ymin + y_size
    zmax = zmin + z_size
    xmin = xmin / scale
    xmax = xmax / scale
    ymin = ymin / scale
    ymax = ymax / scale
    zmin = zmin / scale
    zmax = zmax / scale
    pose_path = os.path.join(rootdir, "test/{:06d}".format(id), "scene_gt.json")
    pose_info = read_pose_np(pose_path)

    for k in range(img_num):
        print(cls_name, k)
        mask_path = os.path.join(rootdir, "test/{:06d}".format(id), "mask_visib", '{:06d}_000000.png'.format(k))
        mask = read_mask_np(mask_path)

        R, t = pose_info[str(k)][0]["cam_R_m2c"], pose_info[str(k)][0]["cam_t_m2c"]
        R = np.array(R)
        R = np.reshape(R, (3, 3))  # 重整一下形状
        t = np.reshape(np.array(t), (3, 1)) / scale
        # 需要把物体的4个坐标截取下来
        xyz_info = os.path.join(rootdir, "test/xyz_crop/{:06d}".format(id), "{:06d}_000000.pkl".format(k))
        xyz = mmcv.load(xyz_info)
        x1, y1, x2, y2 = xyz["xyxy"]
        camK = intrinsic_matrix['linemod'].copy()
        camK_inv = np.linalg.inv(camK)
        # 开始计算遮挡关系
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
        #  生成Q0的坐标
        #  get Q0
        #  show the result
        '''
        pic_point = Q0_z[:, occ_mask_z.astype(np.bool)]
        pic_point = pic_point.T
        plt.figure("3D scatter", facecolor="lightgray")
        ax3d = plt.gca(projection="3d")
        x = pic_point[:, 0]
        y = pic_point[:, 1]
        z = pic_point[:, 2]
        ax3d.scatter(x, y, z, s=20, marker=".", cmap='spectral')
        ax3d.set_xlabel("x label")
        ax3d.set_ylabel("y_label")
        ax3d.set_zlabel("z_label")
        plt.show()
        '''

        Q0 = np.concatenate((Q0_x[1:, :, :], Q0_y[0:1, :, :], Q0_y[2:, :, :], Q0_z[:2, :, :]), axis=0)
        # CHW -  HWC
        Q0 = Q0.transpose((1, 2, 0))
        Q0 = {
            "occ_crop": Q0[y1:y2 + 1, x1:x2 + 1, :],
            "xyxy": [x1, y1, x2, y2],
        }
        #  存储 Q0的坐标
        outpath = os.path.join(rootdir, "test/Q0/{:06d}".format(id), '{:06d}_000000.pkl'.format(k))

        mmcv.dump(Q0, outpath)

def run_lm_q0():
    root_dir = "/data/wanggu/Storage/BOP_DATASETS/lm"
    obj_name = [
        "ape",
        "benchvise",
        "bowl",
        "camera",
        "can",
        "cat",
        "cup",
        "driller",
        "duck",
        "eggbox",
        "glue",
        "holepuncher",
        "iron",
        "lamp",
        "phone"]  # 15个分别处理
    for cls_name in obj_name:
        estimate_occ_mask_Q0(root_dir, cls_name)

if __name__ == "__main__":
    root_dir = "/data/wanggu/Storage/BOP_DATASETS/lm"
    obj_name = [
    "ape",
    "benchvise",
    "bowl",
    "camera",
    "can",
    "cat",
    "cup",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
    "iron",
    "lamp",
    "phone"]  # 15个分别处理
    for cls_name in obj_name:
        estimate_occ_mask_Q0(root_dir, cls_name)
