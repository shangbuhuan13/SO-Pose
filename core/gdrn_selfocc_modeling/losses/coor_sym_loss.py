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


class COOR_loss(nn.Module):
    def __init__(self,
                 loss_type="L2",
                 loss_weight=1.0
                 ):
        super().__init__()
        self.loss_type = "L1"
        self.loss_weight = 1.0

    def forward(self, out_rot, gt_rot,  pred_P0, gt_P0,  pred_Q0, gt_Q0, gt_occmask, gt_mask_xyz, roi_extent,
            gt_transes=None, pred_transes=None):
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
        if out_rot.shape[-1] == 4:
            pred_rots = quat2mat_torch(out_rot)

        # 计算cross task consistency
        '''
        (Rn)^Tt (RP0+t)        =  (Rn)^T(RP0+t) (RQ0+t)
        '''
        #  需要把P0, Q0还原到原来的大小，之前是做了归一化的
        denormalize_P0 = (pred_P0-0.5) * (roi_extent.repeat(h, w, 1, 1).permute(2, 3, 0, 1))
        roi_p0_x = denormalize_P0[:, 0, :, :]  # B, h, w
        roi_p0_y = denormalize_P0[:, 1, :, :]
        roi_p0_z = denormalize_P0[:, 2, :, :]
        # roi_p0 = torch.stack([roi_p0_x, roi_p0_y, roi_p0_z], dim=1)  # B, 3, h, w
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
        # handle gt P0 and Q0
        denormalize_P0 = (gt_P0 - 0.5) * (roi_extent.repeat(h, w, 1, 1).permute(2, 3, 0, 1))
        gt_p0_x = denormalize_P0[:, 0, :, :]  # B, h, w
        gt_p0_y = denormalize_P0[:, 1, :, :]
        gt_p0_z = denormalize_P0[:, 2, :, :]
        # gt_p0 = torch.stack([roi_p0_x, roi_p0_y, roi_p0_z], dim=1)  # B, 3, h, w
        # 处理Q0
        '''
        roi_extent_q0 = torch.stack([roi_extent[:, 1],
                                     roi_extent[:, 2],
                                     roi_extent[:, 0],
                                     roi_extent[:, 2],
                                     roi_extent[:, 0],
                                     roi_extent[:, 1]], dim=1)
        '''
        denormalize_Q0 = (gt_Q0 - 0.5) * (roi_extent_q0.repeat(h, w, 1, 1).permute(2, 3, 0, 1))
        gt_q0_xy_x = denormalize_Q0[:, 0, :, :] # b, h, w
        gt_q0_xy_y = denormalize_Q0[:, 1, :, :]
        gt_q0_xz_x = denormalize_Q0[:, 2, :, :]
        gt_q0_xz_z = denormalize_Q0[:, 3, :, :]
        gt_q0_yz_y = denormalize_Q0[:, 4, :, :]
        gt_q0_yz_z = denormalize_Q0[:, 5, :, :]
        gt_q0_x = torch.stack([torch.zeros([b, h, w], dtype=torch.float).cuda(), roi_q0_xy_x, roi_q0_xy_y],
                               dim=1)  # n=(1,0,0)  # [b, 3, h, w]
        gt_q0_y = torch.stack([roi_q0_xz_x, torch.zeros([b, h, w], dtype=torch.float).cuda(), roi_q0_xz_z], dim=1)
        gt_q0_z = torch.stack([roi_q0_yz_y, roi_q0_yz_z, torch.zeros([b, h, w], dtype=torch.float).cuda()], dim=1)

        # the following four lines are only used for test
        '''
        roi_p0 = pred_P0
        roi_q0_x = pred_Q0[:, 0:2, :, :]
        roi_q0_y = pred_Q0[:, 2:4, :, :]
        roi_q0_z = pred_Q0[:, 4:, :, :]
        '''
        #  可以开始计算了
        pred_outx3D= torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        pred_outy3D= torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        pred_outz3D= torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        gt_P3D= torch.zeros([b, 3, h, w], dtype=torch.float).cuda()
        pred_q_xy_x3D = torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        pred_q_xy_y3D = torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        pred_q_xy_z3D = torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        pred_q_xz_x3D = torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        pred_q_xz_y3D = torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        pred_q_xz_z3D = torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        pred_q_yz_x3D = torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        pred_q_yz_y3D = torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        pred_q_yz_z3D = torch.zeros([b, 1, h, w], dtype=torch.float).cuda()
        gt_Q3D = torch.zeros([b, 9, h, w], dtype=torch.float).cuda()
        for i in range(b):
            pred_R = out_rot[i, :, :].view(3, 3)
            gt_R = gt_rot[i, :, :].view(3, 3)
            out_x_R = pred_R[0, 0] * roi_p0_x[i, :, :] + pred_R[0, 1] * roi_p0_y[i, :, :] \
                      + pred_R[0, 2] * roi_p0_z[i, :, :]
            out_y_R = pred_R[1, 0] * roi_p0_x[i, :, :] + pred_R[1, 1] * roi_p0_y[i, :, :] \
                      + pred_R[1, 2] * roi_p0_z[i, :, :]
            out_z_R = pred_R[2, 0] * roi_p0_x[i, :, :] + pred_R[2, 1] * roi_p0_y[i, :, :] \
                      + pred_R[2, 2] * roi_p0_z[i, :, :]
            out_Q0_xy_x_R = pred_R[0, 1] * roi_q0_xy_x[i, :, :] + pred_R[0, 2] * roi_q0_xy_y[i, :, :]
            out_Q0_xy_y_R = pred_R[1, 1] * roi_q0_xy_x[i, :, :] + pred_R[1, 2] * roi_q0_xy_y[i, :, :]
            out_Q0_xy_z_R = pred_R[2, 1] * roi_q0_xy_x[i, :, :] + pred_R[2, 2] * roi_q0_xy_y[i, :, :]

            out_Q0_xz_x_R = pred_R[0, 0] * roi_q0_xz_x[i, :, :] + pred_R[0, 2] * roi_q0_xz_z[i, :, :]
            out_Q0_xz_y_R = pred_R[1, 0] * roi_q0_xz_x[i, :, :] + pred_R[1, 2] * roi_q0_xz_z[i, :, :]
            out_Q0_xz_z_R = pred_R[2, 0] * roi_q0_xz_x[i, :, :] + pred_R[2, 2] * roi_q0_xz_z[i, :, :]

            out_Q0_yz_x_R = pred_R[0, 0] * roi_q0_yz_y[i, :, :] + pred_R[0, 1] * roi_q0_yz_z[i, :, :]
            out_Q0_yz_y_R = pred_R[1, 0] * roi_q0_yz_y[i, :, :] + pred_R[1, 1] * roi_q0_yz_z[i, :, :]
            out_Q0_yz_z_R = pred_R[2, 0] * roi_q0_yz_y[i, :, :] + pred_R[2, 1] * roi_q0_yz_z[i, :, :]

            gt_x_R = gt_R[0, 0] * gt_p0_x[i, :, :] + gt_R[0, 1] * gt_p0_y[i, :, :] \
                      + gt_R[0, 2] * gt_p0_z[i, :, :]
            gt_y_R = gt_R[1, 0] * gt_p0_x[i, :, :] + gt_R[1, 1] * gt_p0_y[i, :, :] \
                      + gt_R[1, 2] * gt_p0_z[i, :, :]
            gt_z_R = gt_R[2, 0] * gt_p0_x[i, :, :] + gt_R[2, 1] * gt_p0_y[i, :, :] \
                      + gt_R[2, 2] * gt_p0_z[i, :, :]
            gt_Q0_xy_x_R = gt_R[0, 1] * gt_q0_xy_x[i, :, :] + gt_R[0, 2] * gt_q0_xy_y[i, :, :]
            gt_Q0_xy_y_R = gt_R[1, 1] * gt_q0_xy_x[i, :, :] + gt_R[1, 2] * gt_q0_xy_y[i, :, :]
            gt_Q0_xy_z_R = gt_R[2, 1] * gt_q0_xy_x[i, :, :] + gt_R[2, 2] * gt_q0_xy_y[i, :, :]

            gt_Q0_xz_x_R = gt_R[0, 0] * gt_q0_xz_x[i, :, :] + gt_R[0, 2] * gt_q0_xz_z[i, :, :]
            gt_Q0_xz_y_R = gt_R[1, 0] * gt_q0_xz_x[i, :, :] + gt_R[1, 2] * gt_q0_xz_z[i, :, :]
            gt_Q0_xz_z_R = gt_R[2, 0] * gt_q0_xz_x[i, :, :] + gt_R[2, 2] * gt_q0_xz_z[i, :, :]

            gt_Q0_yz_x_R = gt_R[0, 0] * gt_q0_yz_y[i, :, :] + gt_R[0, 1] * gt_q0_yz_z[i, :, :]
            gt_Q0_yz_y_R = gt_R[1, 0] * gt_q0_yz_y[i, :, :] + gt_R[1, 1] * gt_q0_yz_z[i, :, :]
            gt_Q0_yz_z_R = gt_R[2, 0] * gt_q0_yz_y[i, :, :] + gt_R[2, 1] * gt_q0_yz_z[i, :, :]

            # restore the number
            pred_outx3D[i, :, :, :] = out_x_R
            pred_outy3D[i, :, :, :] = out_y_R
            pred_outz3D[i, :, :, :] = out_z_R

            gt_P3D[i, 0, :, :] = gt_x_R
            gt_P3D[i, 1, :, :] = gt_y_R
            gt_P3D[i, 2, :, :] = gt_z_R

            pred_q_xy_x3D[i, :, :, :] = out_Q0_xy_x_R
            pred_q_xy_y3D[i, :, :, :] = out_Q0_xy_y_R
            pred_q_xy_z3D[i, :, :, :] = out_Q0_xy_z_R
            pred_q_xz_x3D[i, :, :, :] = out_Q0_xz_x_R
            pred_q_xz_y3D[i, :, :, :] = out_Q0_xz_y_R
            pred_q_xz_z3D[i, :, :, :] = out_Q0_xz_z_R
            pred_q_yz_x3D[i, :, :, :] = out_Q0_yz_x_R
            pred_q_yz_y3D[i, :, :, :] = out_Q0_yz_y_R
            pred_q_yz_z3D[i, :, :, :] = out_Q0_yz_z_R

            gt_Q3D[i, 0, :, :] = gt_Q0_xy_x_R
            gt_Q3D[i, 1, :, :] = gt_Q0_xy_y_R
            gt_Q3D[i, 2, :, :] = gt_Q0_xy_z_R

            gt_Q3D[i, 3, :, :] = gt_Q0_xz_x_R
            gt_Q3D[i, 4, :, :] = gt_Q0_xz_y_R
            gt_Q3D[i, 5, :, :] = gt_Q0_xz_z_R

            gt_Q3D[i, 6, :, :] = gt_Q0_yz_x_R
            gt_Q3D[i, 7, :, :] = gt_Q0_yz_y_R
            gt_Q3D[i, 8, :, :] = gt_Q0_yz_z_R
        # cal loss
        loss_dict = {}
        loss_func = nn.L1Loss(reduction="sum")
        occmask_x = gt_occmask[:, 0, :, :]
        if occmask_x.sum() < 20:
            loss_Q0_x = 0
        else:
            loss_Q0_x = loss_func(pred_q_xy_x3D * occmask_x[:, None],
                                            gt_Q3D[:, 0:1] * occmask_x[:, None]) + \
                                     loss_func(pred_q_xy_y3D * occmask_x[:, None],
                                               gt_Q3D[:, 1:2] * occmask_x[:, None]) + \
                                     loss_func(pred_q_xy_z3D * occmask_x[:, None],
                                               gt_Q3D[:, 2:3] * occmask_x[:, None])
        occmask_y = gt_occmask[:, 1, :, :]
        if occmask_y.sum() < 20:
            loss_Q0_y = 0
        else:
            loss_Q0_y = loss_func(pred_q_xz_x3D * occmask_y[:, None],
                                               gt_Q3D[:, 3:4] * occmask_y[:, None]) + \
                                     loss_func(pred_q_xz_y3D * occmask_y[:, None],
                                               gt_Q3D[:, 4:5] * occmask_y[:, None]) + \
                                     loss_func(pred_q_xz_z3D * occmask_y[:, None],
                                               gt_Q3D[:, 5:6] * occmask_y[:, None])

        occmask_z = gt_occmask[:, 2, :, :]
        if occmask_z.sum() < 20:
            loss_Q0_z = 0
        else:
            loss_Q0_z = loss_func(pred_q_yz_x3D * occmask_z[:, None],
                                               gt_Q3D[:, 6:7] * occmask_z[:, None]) + \
                                    loss_func(pred_q_yz_y3D * occmask_z[:, None],
                                               gt_Q3D[:, 7:8] * occmask_z[:, None]) + \
                                     loss_func(pred_q_yz_z3D * occmask_z[:, None],
                                               gt_Q3D[:, 8:] * occmask_z[:, None])
        loss_dict["loss_Q0"] = (loss_Q0_x + loss_Q0_y + loss_Q0_z) / gt_occmask.sum().float().clamp(min=1.0)
        # jisuan x
        loss_dict["loss_coor_x"] = loss_func(
            pred_outx3D * gt_mask_xyz[:, None], gt_P3D[:, 0:1] * gt_mask_xyz[:, None]
        ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        loss_dict["loss_coor_y"] = loss_func(
            pred_outy3D * gt_mask_xyz[:, None], gt_P3D[:, 1:2] * gt_mask_xyz[:, None]
        ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        loss_dict["loss_coor_z"] = loss_func(
            pred_outz3D * gt_mask_xyz[:, None], gt_P3D[:, 2:3] * gt_mask_xyz[:, None]
        ) / gt_mask_xyz.sum().float().clamp(min=1.0)

        return loss_dict
