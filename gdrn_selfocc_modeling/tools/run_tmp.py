import numpy as np
import mmcv
from core.gdrn_selfocc_modeling.tools.generate_pbr_Q0 import Q0_generator
from core.gdrn_selfocc_modeling.tools.generate_pbr_Q0_fast import Q0_generator_fast

if __name__ == "__main__":
    '''
    path_1 = "/home/yan/datasets/fortest/000004_000013-Q0.pkl"
    path_2 = "/data/wanggu/Storage/BOP_DATASETS/lm/train_pbr/Q0/000000/000004_000013-Q0.pkl"
    s1 = mmcv.load(path_1)
    s2 = mmcv.load(path_2)
    s_0 = s1["occ_crop"][:, :, 0] - s2["occ_crop"][:, :, 0]
    s_1 = s1["occ_crop"][:, :, 1] - s2["occ_crop"][:, :, 1]
    s_2 = s1["occ_crop"][:, :, 2] - s2["occ_crop"][:, :, 2]
    s_3 = s1["occ_crop"][:, :, 3] - s2["occ_crop"][:, :, 3]
    s_4 = s1["occ_crop"][:, :, 4] - s2["occ_crop"][:, :, 4]
    s_5 = s1["occ_crop"][:, :, 5] - s2["occ_crop"][:, :, 5]
    t_1 = s1["occ_crop"][:, :, 0]
    t_2 = s2["occ_crop"][:, :, 0]
    t=1

    '''
    '''
    root_dir = "/data/wanggu/Storage/BOP_DATASETS/lm/train_pbr"
    model_dir = "/data/wanggu/Storage/BOP_DATASETS/lm/models"
    G_Q = Q0_generator_fast(root_dir, model_dir)
    G_Q.run(scale=1000)
    '''
    # 000008  obj_id:9 xyz_crop[7,11,:]
    '''
    coor = np.array([0.02606, -0.006306, 0.02823], dtype=np.float)  # xyz_crop
    coor2 = np.array([0.02489066, -0.0077247, 0.02943028], dtype=np.float) # the exact gt
    R = np.array([[0.1677168, 0.9850049, 0.04045619], [0.35540748, -0.02213568, -0.9344495],
                  [-0.91954166, 0.17110144, -0.3537907]], dtype=np.float)
    t = np.array([-0.14339037, -0.01479059, 1.7619466], dtype=np.float)
    K = np.array([[572.4114, 0., 325.2611],
                        [0., 573.57043, 242.04899],
                        [0., 0., 1.]], dtype=np.float)
    P = np.matmul(R, coor.reshape(3, 1)) + t.reshape(3, 1)
    p = np.matmul(K, P)
    p = p/P[2]
    p = p[0:2]/p[2]
    print(p)
    '''
    '''
    #coor = np.array([0.03358703, 0.01835209, -0.06660214], dtype=np.float32)
    #coor = np.array([0.02642884, -0.03056015, -0.0673329], dtype=np.float32)

    #coor = np.array([0.03295379, -0.03690532, -0.07103479], dtype=np.float32)
    coor =np.array([ 0.03395513,  0.01807952, -0.06665546], dtype=np.float32)
    coor =np.array([ 0.03905835, -0.02635944, -0.06614584], dtype=np.float32)
    coor = np.array([0.01165416, 0.04863936, 0.03147185], dtype=np.float32)
    K = np.array([[1066.778, 0.0, 312.9869079589844], [0.0, 1067.487, 241.3108977675438],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.array([[-0.19359441,  0.8001863,   0.5676472], [-0.96047807, -0.03662789, -0.27593553],
                  [-0.20000812, -0.5986322,  0.77565217]], dtype=np.float32)
    t = np.array([0.08298882, -0.07580909,  0.7593005], dtype=np.float32)
    P = np.matmul(R, coor.reshape(3, 1)) + t.reshape(3, 1)
    print(P)
    p = np.matmul(K, P)
    p = p / P[2]
    p = p[0:2] / p[2]
    print(p)
    '''
    coor = np.array([-0.03598753, -0.0038088,  -0.00337574],dtype=np.float32)
    coor = np.array([0, -0.02439459,  0.04125579], dtype=np.float32)
    K = np.array([[1066.778, 0.0, 312.9869079589844], [0.0, 1067.487, 241.3108977675438],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.array([[-0.19359441,  0.8001863,   0.5676472 ], [-0.96047807, -0.03662789, -0.27593553], [-0.20000812, -0.5986322,  0.77565217]], dtype=np.float32)
    t = np.array([[ 0.08298882], [-0.07580909], [ 0.7593005 ]], dtype=np.float32)
    P = np.matmul(R, coor.reshape(3, 1)) + t.reshape(3, 1)
    print(P)
    p = np.matmul(K, P)
    p = p / P[2]
    p = p[0:2] / p[2]
    print(p)