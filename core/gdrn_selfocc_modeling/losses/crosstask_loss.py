# 新增加一个cross task的loss
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


class CT_loss(nn.Module):
    def __init__(self,
                 loss_type="L2",
                 loss_weight=1.0
                 ):
        super().__init__()
        self.loss_type = "L1"
        self.loss_weight = 1.0

    def forward(self, pred_rots, pred_P0, pred_Q0, gt_occmask, roi_extent, pred_transes=None):
        """
                pred_rots: [B, 3, 3]
                gt_rots: [B, 3, 3] or [B, 4]
                gt_occmask: [B, 3, h, w]
                pred_p0 : [B, c, h, w]
                pred_transes: [B, 3]
                gt_transes: [B, 3]
                extents: [B, 3]
                roi_extent : [B, 3]  # 用于把缩放的尺度还原回去
                    stores K rotations regarding symmetries, if not symmetric, None
                """
        b, c, h, w = pred_P0.shape
        if pred_rots.shape[-1] == 4:
            pred_rots = quat2mat_torch(pred_rots)

        # 计算cross task consistency
        '''
        (Rn)^Tt (RP0+t)        =  (Rn)^T(RP0+t) (RQ0+t)
        '''
        #  需要把P0, Q0还原到原来的大小，之前是做了归一化的
        denormalize_P0 = (pred_P0-0.5) * (roi_extent.repeat(h, w, 1, 1).permute(2, 3, 0, 1))
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
        denormalize_Q0 = (pred_Q0-0.5) * (roi_extent_q0.repeat(h, w, 1, 1).permute(2, 3, 0, 1))
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

        # the following four lines are only used for test
        '''
        roi_p0 = pred_P0
        roi_q0_x = pred_Q0[:, 0:2, :, :]
        roi_q0_y = pred_Q0[:, 2:4, :, :]
        roi_q0_z = pred_Q0[:, 4:, :, :]
        '''
        #  可以开始计算了
        RTt = (torch.bmm(pred_rots.permute(0, 2, 1), pred_transes.view(b, 3, 1))).squeeze()  # b, 3
        # 将RTt扩展到和points一样的数量，变成b, 3, n
        RTtn = RTt.repeat(h * w, 1, 1).permute(1, 2, 0)  # 这里n是所有点
        t_trans = pred_transes.view(b, 3).repeat(h * w, 1, 1).permute(1, 2, 0)
        RP0t = torch.bmm(pred_rots, roi_p0.view(b, 3, -1)) + t_trans  # b, 3, n， Rp0+t
        # RP0t_norm = torch.norm(RP0t, p=2, dim=1, keepdim=False)  # b, n
        RQ0t1 = torch.bmm(pred_rots, roi_q0_x.view(b, 3, -1)) + t_trans  # b, 3, n
        # RQ0t1_norm = torch.norm(RQ0t1, p=2, dim=2, keepdim=False)  # b, n
        RQ0t2 = torch.bmm(pred_rots, roi_q0_y.view(b, 3, -1)) + t_trans  # b, 3, n
        # RQ0t2_norm = torch.norm(RQ0t2, p=2, dim=2, keepdim=False)
        RQ0t3 = torch.bmm(pred_rots, roi_q0_z.view(b, 3, -1)) + t_trans  # b, 3, n
        # RQ0t3_norm = torch.norm(RQ0t3, p=2, dim=2, keepdim=False)
        # 开始计算loss
        # 针对于n=(1,0,0)的loss
        loss_x = RTtn[:, 0:1, :].repeat(1, 3, 1) * RP0t - \
                 (torch.bmm(pred_rots[:, :, 0:1].permute(0, 2, 1), RP0t)).repeat(1, 3, 1) * RQ0t1  # 得到b, 3, n
        loss_x = torch.norm(loss_x, dim=1) # b, n
        # b, n * b, n---b, n * b, n  n-h*w
        # 用occmask选择需要计算的点 b,h,w -->b,n
        occmask_x = gt_occmask[:, 0, :, :].view(b, -1)
        loss_x = loss_x * occmask_x  # b,n
        if occmask_x.sum() < b * 3:  # 整个batch
            loss_x = 0
        else:
            loss_x = loss_x.sum()  # 取均值
        # 针对于n=(0,1,0)的loss
        loss_y = RTtn[:, 1:2, :].repeat(1, 3, 1) * RP0t - \
                 (torch.bmm(pred_rots[:, :, 1:2].permute(0, 2, 1), RP0t)).repeat(1, 3, 1) * RQ0t2  # 得到b, 3, n
        loss_y = torch.norm(loss_y, dim=1)
        # b, n * b, n---b, n * b, n  n-h*w
        # 用occmask选择需要计算的点 b,h,w -->b,n
        occmask_y = gt_occmask[:, 1, :, :].view(b, -1)
        loss_y = loss_y * occmask_y  # b,n
        if occmask_y.sum() < b * 3:  # 整个batch
            loss_y = 0
        else:
            loss_y = loss_y.sum()  # 取均值
        # 针对于n=(0,0,1)的loss
        loss_z = RTtn[:, 2:, :].repeat(1, 3, 1) * RP0t - \
                 (torch.bmm(pred_rots[:, :, 2:].permute(0, 2, 1), RP0t)).repeat(1, 3, 1) * RQ0t3  # 得到b, 3, n

        loss_z = torch.norm(loss_z, dim=1)
        # b, n * b, n---b, n * b, n  n-h*w
        # 用occmask选择需要计算的点 b,h,w -->b,n
        occmask_z = gt_occmask[:, 2, :, :].view(b, -1)
        loss_z = loss_z * occmask_z  # b,n
        if occmask_z.sum() < b * 3:  # 整个batch
            loss_z = 0
        else:
            loss_z = loss_z.sum()   # 取均值
        # 最终的loss
        loss = (loss_x + loss_y + loss_z)/gt_occmask.sum().float().clamp(min=1.0)
        return loss
