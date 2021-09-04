import sys
import os.path as osp
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.pose_utils import quat2mat_torch
from .l2_loss import L2Loss
from fvcore.nn import smooth_l1_loss
from lib.utils.utils import dprint
from core.utils.pose_utils import get_closest_rot_batch
import logging
from detectron2.utils.logger import log_first_n
from lib.pysixd.misc import transform_pts_batch

logger = logging.getLogger(__name__)


# 这里的loss主要是和q0有关的两个， q0-p0, q0-u
class CT_loss_projection(nn.Module):
    def __init__(self,
                 loss_type="L2",
                 loss_weight=1.0
                 ):
        super().__init__()
        self.loss_type = "L1"
        self.loss_weight = 1.0

    def P02P(self, rots, trans, P0):
        b, c, n = P0.shape
        t = trans.view(b, 3, 1).repeat(1, 1, n)
        P = torch.bmm(rots, P0) + t
        return P

    # 其中K是相机内参矩阵，3x3
    # roi_2d是每个p0对应的2d坐标，在dataloader里面拿进来的
    def forward(self, pred_rots, pred_P0, pred_Q0, gt_occmask, roi_extent, pred_transes, roi_2d, imH, imW, K=None):
            b, c, h, w = pred_P0.shape
            # need -1
            imW = imW - 1
            imH = imH - 1
            BimSize = torch.cat([imW.view(-1, 1), imH.view(-1, 1)], dim=1)
            # 生成2D坐标的阵列,roi2d [0,1]--->[0, w-1]
            roi_2d = roi_2d * (BimSize.repeat(h, w, 1, 1).permute(2, 3, 0, 1))
            if pred_rots.shape[-1] == 4:
                pred_rots = quat2mat_torch(pred_rots)

            # P, Q and camera optic lie on the same line

            #  需要把P0, Q0还原到原来的大小，之前是做了归一化的
            denormalize_P0 = (pred_P0 - 0.5) * (roi_extent.repeat(h, w, 1, 1).permute(2, 3, 0, 1))
            roi_p0_x = denormalize_P0[:, 0, :, :]  # B, h, w
            roi_p0_y = denormalize_P0[:, 1, :, :]
            roi_p0_z = denormalize_P0[:, 2, :, :]
            roi_p0 = torch.stack([roi_p0_x, roi_p0_y, roi_p0_z], dim=1)  # B, 3, h, w
            # 处理Q0
            roi_extent_q0 = torch.stack([roi_extent[:, 1],
                                         roi_extent[:, 2],
                                         roi_extent[:, 0],
                                         roi_extent[:, 2],
                                         roi_extent[:, 0],
                                         roi_extent[:, 1]], dim=1)
            denormalize_Q0 = (pred_Q0 - 0.5) * (roi_extent_q0.repeat(h, w, 1, 1).permute(2, 3, 0, 1))
            roi_q0_xy_x = denormalize_Q0[:, 0, :, :]
            roi_q0_xy_y = denormalize_Q0[:, 1, :, :]
            roi_q0_xz_x = denormalize_Q0[:, 2, :, :]
            roi_q0_xz_z = denormalize_Q0[:, 3, :, :]
            roi_q0_yz_y = denormalize_Q0[:, 4, :, :]
            roi_q0_yz_z = denormalize_Q0[:, 5, :, :]
            roi_q0_x = torch.stack([torch.zeros([b, h, w], dtype=torch.float).cuda(), roi_q0_xy_x, roi_q0_xy_y],
                                   dim=1)  # n=(1,0,0)  # [b, 3, h, w]
            roi_q0_y = torch.stack([roi_q0_xz_x, torch.zeros([b, h, w], dtype=torch.float).cuda(), roi_q0_xz_z], dim=1)
            roi_q0_z = torch.stack([roi_q0_yz_y, roi_q0_yz_z, torch.zeros([b, h, w], dtype=torch.float).cuda()], dim=1)
            z_mask_sum = torch.zeros([3], dtype=torch.float).cuda()
            if K.numel() > 9:  # 这里K的形状是b, 3, 3
                # 针对p0进行预计算
                '''
                XK = torch.tensor([572.4114, 0.0, 325.2611, 0.0, 573.5704, 242.0489, 0.0, 0.0, 1.0], dtype=torch.float32).view(3, 3)
                XK = XK.repeat(b, 1, 1).cuda()
                '''
                roi_p0 = roi_p0.view(b, 3, -1)  # b, 3, n
                P = self.P02P(pred_rots, pred_transes, roi_p0) # RP0t
                p0_projection = (torch.bmm(K, P)).permute(0, 2, 1)  # b, n, 3, k(rR0t+)
                # 针对q0_x的计算
                roi_q0_x = roi_q0_x.view(b, 3, -1)  # b, 3,n
                Q_x = self.P02P(pred_rots, pred_transes, roi_q0_x) # b, 3, n
                occmask_x = gt_occmask[:, 0, :, :].view(b, -1).bool() # b,n
                q0_x_projection = (torch.bmm(K, Q_x)).permute(0, 2, 1) # b,n,3
                z_mask = (torch.abs(P[:, 2, :]) > 1e-4) & (torch.abs(Q_x[:, 2, :]) > 1e-4) & occmask_x # b, n
                z_mask_sum[0] = z_mask.sum()
                if z_mask.sum() < b * 3:
                    loss_x = 0
                else:
                    q0_pro_x = q0_x_projection[z_mask, :]  # m, 3
                    p0_pro_x = p0_projection[z_mask, :]  # m, 3
                    u_x = roi_2d.view(b, 2, -1).permute(0, 2, 1)  # b, n, 2
                    u_x = u_x[z_mask, :] # m,2
                    q0_x_norm = (Q_x[:, 2, :][z_mask]).view(-1, 1) # m, 1
                    p0_x_norm = (P[:, 2, :][z_mask]).view(-1, 1)  # m, 1
                    p0_n_x = p0_pro_x[:, 0:2] / p0_x_norm  # 除以zp
                    q0_n_x = q0_pro_x[:, 0:2] / q0_x_norm
                    loss_x_pq = p0_n_x - q0_n_x  # m, 2
                    loss_x_qu = q0_n_x - u_x  # m, 2
                    if self.loss_type is "L1":
                        loss_x = torch.sum(torch.abs(loss_x_qu)) + torch.sum(torch.abs(loss_x_pq))
                    else:   # L2
                        loss_x = torch.sum(torch.norm(loss_x_pq, dim=1)) + torch.sum(torch.norm(loss_x_qu, dim=1))

                # 针对q0_y的计算
                roi_q0_y = roi_q0_y.view(b, 3, -1)  # b, 3,n
                Q_y = self.P02P(pred_rots, pred_transes, roi_q0_y) # b, 3, n
                occmask_y = gt_occmask[:, 1, :, :].view(b, -1).bool() # b,n
                q0_y_projection = (torch.bmm(K, Q_y)).permute(0, 2, 1) # b,n,3
                z_mask = (torch.abs(P[:, 2, :]) > 1e-4) & (torch.abs(Q_y[:, 2, :]) > 1e-4) & occmask_y # b, n
                z_mask_sum[1] = z_mask.sum()
                if z_mask.sum() < b * 3:
                    loss_y = 0
                else:
                    q0_pro_y = q0_y_projection[z_mask, :]  # m, 3
                    p0_pro_y = p0_projection[z_mask, :]  # m, 3
                    u_y = roi_2d.view(b, 2, -1).permute(0, 2, 1)  # b, n, 2
                    u_y = u_y[z_mask, :] # m,2
                    q0_y_norm = (Q_y[:, 2, :][z_mask]).view(-1, 1) # m, 1
                    p0_y_norm = (P[:, 2, :][z_mask]).view(-1, 1)  # m, 1
                    p0_n_y = p0_pro_y[:, 0:2] / p0_y_norm  # 除以zp
                    q0_n_y = q0_pro_y[:, 0:2] / q0_y_norm
                    loss_y_pq = p0_n_y - q0_n_y  # m, 2
                    loss_y_qu = q0_n_y - u_y  # m, 2
                    if self.loss_type is "L1":
                        loss_y = torch.sum(torch.abs(loss_y_qu)) + torch.sum(torch.abs(loss_y_pq))
                    else:   # L2
                        loss_y = torch.sum(torch.norm(loss_y_pq, dim=1)) + torch.sum(torch.norm(loss_y_qu, dim=1))

                # 针对q0_z的计算
                roi_q0_z = roi_q0_z.view(b, 3, -1)  # b, 3,n
                Q_z = self.P02P(pred_rots, pred_transes, roi_q0_z)  # b, 3, n
                occmask_z = gt_occmask[:, 2, :, :].view(b, -1).bool()  # b,n
                q0_z_projection = (torch.bmm(K, Q_z)).permute(0, 2, 1)  # b,n,3
                z_mask = (torch.abs(P[:, 2, :]) > 1e-4) & (torch.abs(Q_z[:, 2, :]) > 1e-4) & occmask_z  # b, n
                z_mask_sum[2] = z_mask.sum()
                if z_mask.sum() < b * 3:
                    loss_z = 0
                else:
                    q0_pro_z = q0_z_projection[z_mask, :]  # m, 3
                    p0_pro_z = p0_projection[z_mask, :]  # m, 3
                    u_z = roi_2d.view(b, 2, -1).permute(0, 2, 1)  # b, n, 2
                    u_z = u_z[z_mask, :]  # m,2
                    q0_z_norm = (Q_z[:, 2, :][z_mask]).view(-1, 1)  # m, 1
                    p0_z_norm = (P[:, 2, :][z_mask]).view(-1, 1)  # m, 1
                    p0_n_z = p0_pro_z[:, 0:2] / p0_z_norm  # 除以zp
                    q0_n_z = q0_pro_z[:, 0:2] / q0_z_norm
                    loss_z_pq = p0_n_z - q0_n_z  # m, 2
                    loss_z_qu = q0_n_z - u_z  # m, 2
                    if self.loss_type is "L1":
                        loss_z = torch.sum(torch.abs(loss_z_qu)) + torch.sum(torch.abs(loss_z_pq))
                    else:  # L2
                        loss_z = torch.sum(torch.norm(loss_z_pq, dim=1)) + torch.sum(torch.norm(loss_z_qu, dim=1))
                loss = ((loss_x + loss_y + loss_z) / (z_mask_sum.sum())) / 572.3   # depends on K
            else:
                loss = 0  # 这个loss不要就是了
            return loss

'''
# 计算投影点
def projection(P, K, type="batch"):
    if type == "batch":  # 对应计算一个batch的投影点，否则计算一个点的投影点，此时P的大小为b, 3, n
        if K.numel() > 9:  # K本身是b,3,3
            u = torch.bmm(K, P) / (P[:, 2:, :].repeat(1, 3, 1))  # b,3,n
            # 转化为非齐次坐标
            u = u[:, 0:2, :] / u[:, 2:, :].repeat(1, 3, 1)  # b, 2, n
        else:  # K本身是3， 3
            u = K[0, 0] * P[:, 0:1, :] / P[:, 2:, :] + K[0, 2]  # b, 1, n
            v = K[1, 1] * P[:, 1:2, :] / P[:, 2:, :] + K[1, 2]
            u = torch.cat([u, v], dim=1)
        return u
    else:  # 只计算一个点P, 3, 此时K3， 3
        P = P.squeeze()
        u = K[0, 0] * P[0] / P[2] + K[0, 2]
        v = K[1, 1] * P[1] / P[2] + K[1, 2]
        u = torch.cat([u, v], dim=0)  # size为2
        return u
'''