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

class Q_def_loss(nn.Module):
    def __init__(self, loss_type="L2", loss_weight=1.0):
        super().__init__()
        self.loss_type = "L1"
        self.loss_weight = 1.0
    def Q02Q(self, rots, trans, Q0):  # 返回RQ0+t
        b, c, n = Q0.shape
        t = trans.view(b, 3, 1).repeat(1, 1, n)
        Q = torch.bmm(rots, Q0) + t
        return Q
    def forward(self, rots, pred_Q0, gt_occmask, roi_extent, transes, roi_2d, imH, imW, K=None):
        b, c, h, w = pred_Q0.shape
        # need -1
        imW = imW - 1
        imH = imH - 1
        BimSize = torch.cat([imW.view(-1, 1), imH.view(-1, 1)], dim=1)
        # 生成2D坐标的阵列,roi2d [0,1]--->[0, w-1]
        roi_2d = roi_2d * (BimSize.repeat(h, w, 1, 1).permute(2, 3, 0, 1))
        if rots.shape[-1] == 4:
            pred_rots = quat2mat_torch(rots)
        #  需要把Q0还原到原来的大小，之前是做了归一化的
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

        if K.numel() > 9:
            # 针对q0_x的计算
            roi_q0_x = roi_q0_x.view(b, 3, -1)  # b, 3,n
            Q_x = self.Q02Q(rots, transes, roi_q0_x)  # b, 3, n
            occmask_x = gt_occmask[:, 0, :, :].view(b, -1).bool()  # b,n
            q0_x_projection = (torch.bmm(K, Q_x)).permute(0, 2, 1)  # b,n,3
            z_mask = (torch.abs(Q_x[:, 2, :]) > 1e-4) & occmask_x  # b, n
            if z_mask.sum() < b * 3:
                loss_x = 0
            else:
                q0_pro_x = q0_x_projection[z_mask, :]  # m, 3
                u_x = roi_2d.view(b, 2, -1).permute(0, 2, 1)  # b, n, 2
                u_x = u_x[z_mask, :]  # m,2
                q0_x_norm = (Q_x[:, 2, :][z_mask]).view(-1, 1)  # m, 1
                q0_n_x = q0_pro_x[:, 0:2] / q0_x_norm
                loss_x_qu = q0_n_x - u_x  # m, 2
                if self.loss_type is "L1":
                    loss_x = torch.mean(torch.abs(loss_x_qu))
                else:  # L2
                    loss_x = torch.mean(torch.norm(loss_x_qu, dim=1))

            # 针对q0_y的计算
            roi_q0_y = roi_q0_y.view(b, 3, -1)  # b, 3,n
            Q_y = self.Q02Q(rots, transes, roi_q0_y)  # b, 3, n
            occmask_y = gt_occmask[:, 1, :, :].view(b, -1).bool()  # b,n
            q0_y_projection = (torch.bmm(K, Q_y)).permute(0, 2, 1)  # b,n,3
            z_mask = (torch.abs(Q_y[:, 2, :]) > 1e-4) & occmask_y  # b, n
            if z_mask.sum() < b * 3:
                loss_y = 0
            else:
                q0_pro_y = q0_y_projection[z_mask, :]  # m, 3
                u_y = roi_2d.view(b, 2, -1).permute(0, 2, 1)  # b, n, 2
                u_y = u_y[z_mask, :]  # m,2
                q0_y_norm = (Q_y[:, 2, :][z_mask]).view(-1, 1)  # m, 1
                q0_n_y = q0_pro_y[:, 0:2] / q0_y_norm
                loss_y_qu = q0_n_y - u_y  # m, 2
                if self.loss_type is "L1":
                    loss_y = torch.mean(torch.abs(loss_y_qu))
                else:  # L2
                    loss_y = torch.mean(torch.norm(loss_y_qu, dim=1))
            # 针对Q_z的计算
            # 针对q0_z的计算
            roi_q0_z = roi_q0_z.view(b, 3, -1)  # b, 3,n
            Q_z = self.Q02Q(rots, transes, roi_q0_z)  # b, 3, n
            occmask_z = gt_occmask[:, 2, :, :].view(b, -1).bool()  # b,n
            q0_z_projection = (torch.bmm(K, Q_z)).permute(0, 2, 1)  # b,n,3
            z_mask = (torch.abs(Q_z[:, 2, :]) > 1e-4) & occmask_z  # b, n
            if z_mask.sum() < b * 3:
                loss_z = 0
            else:
                q0_pro_z = q0_z_projection[z_mask, :]  # m, 3

                u_z = roi_2d.view(b, 2, -1).permute(0, 2, 1)  # b, n, 2
                u_z = u_z[z_mask, :]  # m,2
                q0_z_norm = (Q_z[:, 2, :][z_mask]).view(-1, 1)  # m, 1
                q0_n_z = q0_pro_z[:, 0:2] / q0_z_norm
                loss_z_qu = q0_n_z - u_z  # m, 2
                if self.loss_type is "L1":
                    loss_z = torch.mean(torch.abs(loss_z_qu))
                else:  # L2
                    loss_z = torch.mean(torch.norm(loss_z_qu, dim=1))

            loss = (loss_x + loss_y + loss_z) / 572.5
        else:
            loss = 0
        return loss
