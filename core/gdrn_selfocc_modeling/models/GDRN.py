import copy
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.solver_utils import build_optimizer_with_params
from detectron2.utils.events import get_event_storage
from mmcv.runner import load_checkpoint
import mmcv
from ..losses.coor_cross_entropy import CrossEntropyHeatmapLoss
from ..losses.l2_loss import L2Loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss
from ..losses.crosstask_loss import CT_loss
from ..losses.crosstask_projection_loss import CT_loss_projection
from ..losses.q0_ref_loss import Q_def_loss
from ..losses.coor_sym_loss import COOR_loss
from .model_utils import (
    compute_mean_re_te,
    get_neck,
    get_geo_head,
    get_selfocc_head,
    get_mask_prob,
    get_pnp_net,
    get_rot_mat,
    get_xyz_mask_region_out_dim,
    get_selfocc_out_dim,
)
from .pose_from_pred import pose_from_pred
from .pose_from_pred_centroid_z import pose_from_pred_centroid_z
from .pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from .net_factory import BACKBONES

logger = logging.getLogger(__name__)


class GDRN(nn.Module):
    def __init__(self, cfg, backbone, geo_head_net, selfocc_head_net, neck=None, pnp_net=None):
        super().__init__()
        assert cfg.MODEL.POSE_NET.NAME == "GDRN", cfg.MODEL.POSE_NET.NAME
        self.backbone = backbone
        self.neck = neck

        self.epoch_flag = 30
        self.meanre_save = 0
        self.meante_save = 0
        self.epoch_count = 0

        self.geo_head_net = geo_head_net
        self.selfocc_head_net = selfocc_head_net
        self.pnp_net = pnp_net

        self.cfg = cfg
        self.xyz_out_dim, self.mask_out_dim, self.region_out_dim = get_xyz_mask_region_out_dim(cfg)
        self.Q0_out_dim, self.occmask_out_dim = get_selfocc_out_dim(cfg)
        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        # yapf: disable
        if cfg.MODEL.POSE_NET.USE_MTL:
            self.loss_names = [
                "mask", "coor_x", "coor_y", "coor_z", "coor_x_bin", "coor_y_bin", "coor_z_bin", "region",
                "PM_R", "PM_xy", "PM_z", "PM_xy_noP", "PM_z_noP", "PM_T", "PM_T_noP",
                "centroid", "z", "trans_xy", "trans_z", "trans_LPnP", "rot", "bind",
            ]
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32))
                )
        # yapf: enable

    def forward(
        self,
        x,
        gt_xyz=None,
        gt_xyz_bin=None,
        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_mask_obj=None,
        gt_mask_erode=None,
        gt_region=None,
        gt_ego_rot=None,
        gt_points=None,
        sym_infos=None,
        gt_trans=None,
        gt_trans_ratio=None,
        roi_classes=None,
        roi_coord_2d=None,
        roi_cams=None,
        roi_centers=None,
        roi_whs=None,
        roi_extents=None,
        resize_ratios=None,
        do_loss=False,
        # selfocc
        gt_q0=None,
        gt_occmask=None,
        roi_extent=None,
        size_imW=None,
        size_imH=None,
        E_step=None,

    ):
        cfg = self.cfg
        net_cfg = cfg.MODEL.POSE_NET
        g_head_cfg = net_cfg.GEO_HEAD
        s_head_cfg = net_cfg.SELFOCC_HEAD
        pnp_net_cfg = net_cfg.PNP_NET

        device = x.device
        bs = x.shape[0]
        num_classes = net_cfg.NUM_CLASSES
        out_res = net_cfg.OUTPUT_RES

        # x.shape [bs, 3, 256, 256]
        conv_feat = self.backbone(x)  # [bs, c, 8, 8]
        if self.neck is not None:
            conv_feat = self.neck(conv_feat)
        mask, coor_x, coor_y, coor_z, region = self.geo_head_net(conv_feat)
        if s_head_cfg.OCCMASK_AWARE:
            occmask, Q0_xy_x, Q0_xy_y, Q0_xz_x, Q0_xz_z, Q0_yz_y, Q0_yz_z = self.selfocc_head_net(conv_feat)
        else:
            Q0_xy_x, Q0_xy_y, Q0_xz_x, Q0_xz_z, Q0_yz_y, Q0_yz_z = self.selfocc_head_net(conv_feat)
        '''
        save_x = x.detach().cpu().numpy()
        save_coor = torch.cat([coor_x, coor_y, coor_z], dim=1).detach().cpu().numpy()
        save_occ = torch.cat([Q0_xy_x, Q0_xy_y, Q0_xz_x, Q0_xz_z, Q0_yz_y, Q0_yz_z], dim=1).detach().cpu().numpy()
        save_gt_occ = gt_q0.detach().cpu().numpy()
        save_gt_coor = gt_xyz.detach().cpu().numpy()
        save_gt_mask = gt_mask_visib.detach().cpu().numpy()
        save_gt_occmask = gt_occmask.detach().cpu().numpy()
        save_pred_mask = mask.detach().cpu().numpy()
        outpath = "/home/yan/gdnr_selfocc/core/gdrn_selfocc_modeling/output/gdrn_selfocc/ycbv/x.pkl"
        outpath2 = "/home/yan/gdnr_selfocc/core/gdrn_selfocc_modeling/output/gdrn_selfocc/ycbv/coor.pkl"
        outpath3 = "/home/yan/gdnr_selfocc/core/gdrn_selfocc_modeling/output/gdrn_selfocc/ycbv/occ.pkl"
        outpath4 = "/home/yan/gdnr_selfocc/core/gdrn_selfocc_modeling/output/gdrn_selfocc/ycbv/gt_occ.pkl"
        outpath5 = "/home/yan/gdnr_selfocc/core/gdrn_selfocc_modeling/output/gdrn_selfocc/ycbv/gt_coor.pkl"
        outpath6 = "/home/yan/gdnr_selfocc/core/gdrn_selfocc_modeling/output/gdrn_selfocc/ycbv/gt_mask.pkl"
        outpath7 = "/home/yan/gdnr_selfocc/core/gdrn_selfocc_modeling/output/gdrn_selfocc/ycbv/gt_occmask.pkl"
        outpath8 = "/home/yan/gdnr_selfocc/core/gdrn_selfocc_modeling/output/gdrn_selfocc/ycbv/pred_mask.pkl"
        mmcv.dump(save_x, outpath)
        mmcv.dump(save_coor, outpath2)
        mmcv.dump(save_occ, outpath3)
        mmcv.dump(save_gt_occ, outpath4)
        mmcv.dump(save_gt_coor, outpath5)
        mmcv.dump(save_gt_mask, outpath6)
        mmcv.dump(save_gt_occmask, outpath7)
        mmcv.dump(save_pred_mask, outpath8)
        '''
        if g_head_cfg.XYZ_CLASS_AWARE:
            assert roi_classes is not None
            coor_x = coor_x.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_x = coor_x[torch.arange(bs).to(device), roi_classes]
            coor_y = coor_y.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_y = coor_y[torch.arange(bs).to(device), roi_classes]
            coor_z = coor_z.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_z = coor_z[torch.arange(bs).to(device), roi_classes]

        if s_head_cfg.Q0_CLASS_AWARE:
            Q0_xy_x = Q0_xy_x.view(bs, num_classes, self.Q0_out_dim // 6, out_res, out_res)
            Q0_xy_x = Q0_xy_x[torch.arange(bs).to(device), roi_classes]
            Q0_xz_x = Q0_xz_x.view(bs, num_classes, self.Q0_out_dim // 6, out_res, out_res)
            Q0_xz_x = Q0_xz_x[torch.arange(bs).to(device), roi_classes]
            Q0_yz_y = Q0_yz_y.view(bs, num_classes, self.Q0_out_dim // 6, out_res, out_res)
            Q0_yz_y = Q0_yz_y[torch.arange(bs).to(device), roi_classes]

            Q0_xy_y = Q0_xy_y.view(bs, num_classes, self.Q0_out_dim // 6, out_res, out_res)
            Q0_xy_y = Q0_xy_y[torch.arange(bs).to(device), roi_classes]
            Q0_xz_z = Q0_xz_z.view(bs, num_classes, self.Q0_out_dim // 6, out_res, out_res)
            Q0_xz_z = Q0_xz_z[torch.arange(bs).to(device), roi_classes]
            Q0_yz_z = Q0_yz_z.view(bs, num_classes, self.Q0_out_dim // 6, out_res, out_res)
            Q0_yz_z = Q0_yz_z[torch.arange(bs).to(device), roi_classes]

        if g_head_cfg.MASK_CLASS_AWARE:
            assert roi_classes is not None
            mask = mask.view(bs, num_classes, self.mask_out_dim, out_res, out_res)
            mask = mask[torch.arange(bs).to(device), roi_classes]

        if g_head_cfg.REGION_CLASS_AWARE:
            assert roi_classes is not None
            region = region.view(bs, num_classes, self.region_out_dim, out_res, out_res)
            region = region[torch.arange(bs).to(device), roi_classes]

        # -----------------------------------------------
        # get rot and trans from pnp_net
        # NOTE: use softmax for bins (the last dim is bg)
        if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:
            coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)
            coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
            coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
            Q0_xy_x_sofrmax = F.softmax(Q0_xy_x[:, :-1, :, :])
            Q0_xy_y_sofrmax = F.softmax(Q0_xy_y[:, :-1, :, :])
            Q0_xz_x_sofrmax = F.softmax(Q0_xz_x[:, :-1, :, :])
            Q0_xz_z_sofrmax = F.softmax(Q0_xz_z[:, :-1, :, :])
            Q0_yz_y_sofrmax = F.softmax(Q0_yz_y[:, :-1, :, :])
            Q0_yz_z_sofrmax = F.softmax(Q0_yz_z[:, :-1, :, :])
            coor_feat = torch.cat([coor_x_softmax, coor_y_softmax, coor_z_softmax, Q0_xy_x_sofrmax, Q0_xy_y_sofrmax,
                                   Q0_xz_x_sofrmax, Q0_xz_z_sofrmax, Q0_yz_y_sofrmax, Q0_yz_z_sofrmax], dim=1)
        else:
            coor_feat = torch.cat([coor_x, coor_y, coor_z, Q0_xy_x, Q0_xy_y, Q0_xz_x, Q0_xz_z, Q0_yz_y, Q0_yz_z], dim=1)
            # BCHW
        if pnp_net_cfg.WITH_2D_COORD:
            assert roi_coord_2d is not None
            coor_feat = torch.cat([coor_feat, roi_coord_2d], dim=1)

        # NOTE: for region, the 1st dim is bg
        region_softmax = F.softmax(region[:, 1:, :, :], dim=1)

        mask_atten = None
        if pnp_net_cfg.MASK_ATTENTION != "none":
            mask_atten = get_mask_prob(mask, mask_loss_type=net_cfg.LOSS_CFG.MASK_LOSS_TYPE)

        region_atten = None
        if pnp_net_cfg.REGION_ATTENTION:
            region_atten = region_softmax

        pred_rot_, pred_t_ = self.pnp_net(
            coor_feat, region=region_atten, extents=roi_extents, mask_attention=mask_atten
        )

        # convert pred_rot to rot mat -------------------------
        rot_type = pnp_net_cfg.ROT_TYPE
        pred_rot_m = get_rot_mat(pred_rot_, rot_type)

        # convert pred_rot_m and pred_t to ego pose -----------------------------
        if pnp_net_cfg.TRANS_TYPE == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in rot_type,
                z_type=pnp_net_cfg.Z_TYPE,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "centroid_z_abs":
            # abs 2d obj center and abs z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z_abs(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                eps=1e-4,
                is_allo="allo" in rot_type,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "trans":
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m, pred_t_, eps=1e-4, is_allo="allo" in rot_type, is_train=do_loss
            )
        else:
            raise ValueError(f"Unknown trans type: {pnp_net_cfg.TRANS_TYPE}")

        if not do_loss:  # test
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            if cfg.TEST.USE_PNP:
                # TODO: move the pnp/ransac inside forward
                out_dict.update({"mask": mask, "coor_x": coor_x, "coor_y": coor_y, "coor_z": coor_z, "region": region})
        else:
            out_dict = {}
            assert (
                (gt_xyz is not None)
                and (gt_trans is not None)
                and (gt_trans_ratio is not None)
                and (gt_region is not None)
            )
            mean_re, mean_te = compute_mean_re_te(pred_trans, pred_rot_m, gt_trans, gt_ego_rot)
            vis_dict = {
                "vis/error_R": mean_re,
                "vis/error_t": mean_te * 100,  # cm
                "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
                "vis/tx_pred": pred_trans[0, 0].detach().item(),
                "vis/ty_pred": pred_trans[0, 1].detach().item(),
                "vis/tz_pred": pred_trans[0, 2].detach().item(),
                "vis/tx_net": pred_t_[0, 0].detach().item(),
                "vis/ty_net": pred_t_[0, 1].detach().item(),
                "vis/tz_net": pred_t_[0, 2].detach().item(),
                "vis/tx_gt": gt_trans[0, 0].detach().item(),
                "vis/ty_gt": gt_trans[0, 1].detach().item(),
                "vis/tz_gt": gt_trans[0, 2].detach().item(),
                "vis/tx_rel_gt": gt_trans_ratio[0, 0].detach().item(),
                "vis/ty_rel_gt": gt_trans_ratio[0, 1].detach().item(),
                "vis/tz_rel_gt": gt_trans_ratio[0, 2].detach().item(),
            }
            if E_step is not None:
                if E_step > self.epoch_flag:
                    print("training loss: ",  "mean_re: ", self.meanre_save / self.epoch_count, " mean_te",
                          self.meante_save / self.epoch_count)
                    self.epoch_flag = E_step
                    self.epoch_count = 0
                    self.meante_save = 0
                    self.meanre_save = 0
                else:
                    self.meante_save += mean_te
                    self.meanre_save += mean_re
                    self.epoch_count += 1

            loss_dict = self.gdrn_loss(
                cfg=self.cfg,
                out_mask=mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
                gt_mask_erode=gt_mask_erode,
                out_x=coor_x,
                out_y=coor_y,
                out_z=coor_z,
                gt_xyz=gt_xyz,
                gt_xyz_bin=gt_xyz_bin,
                out_region=region,
                gt_region=gt_region,
                out_trans=pred_trans,
                gt_trans=gt_trans,
                out_rot=pred_ego_rot,
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],  # TODO: get these from trans head
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                extents=roi_extents,
                # roi_classes=roi_classes,
                # selfocc
                gt_occmask=gt_occmask,
                gt_Q0=gt_q0,
                out_Q0_xy_x=Q0_xy_x, out_Q0_xy_y=Q0_xy_y, out_Q0_xz_x=Q0_xz_x,
                out_Q0_xz_z=Q0_xz_z, out_Q0_yz_y=Q0_yz_y, out_Q0_yz_z=Q0_yz_z,
                roi_extent=roi_extent, roi_2d=roi_coord_2d, roi_cam=roi_cams,
                sizeimH=size_imH,
                sizeimW=size_imW,
                E_step=E_step,
            )

            '''
            # only for test
            loss_dict = self.gdrn_loss(
                cfg=self.cfg,
                out_mask=mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
                gt_mask_erode=gt_mask_erode,
                out_x=gt_xyz[:, 0:1, :, :],
                out_y=gt_xyz[:, 1:2, :, :],
                out_z=gt_xyz[:, 2:, :, :],
                gt_xyz=gt_xyz,
                gt_xyz_bin=gt_xyz_bin,
                out_region=region,
                gt_region=gt_region,
                out_trans=gt_trans,
                gt_trans=gt_trans,
                out_rot=gt_ego_rot,
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],  # TODO: get these from trans head
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                extents=roi_extents,
                # roi_classes=roi_classes,
                # selfocc
                gt_occmask=gt_occmask,
                gt_Q0=gt_q0,
                out_Q0_xy_x=gt_q0[:, 0:1, :, :], out_Q0_xy_y=gt_q0[:, 1:2, :, :], out_Q0_xz_x=gt_q0[:, 2:3, :, :],
                out_Q0_xz_z=gt_q0[:, 3:4, :, :], out_Q0_yz_y=gt_q0[:, 4:5, :, :], out_Q0_yz_z=gt_q0[:, 5:, :, :],
                roi_extent=roi_extent, roi_2d=roi_coord_2d, roi_cam=roi_cams,
                sizeimH=size_imH,
                sizeimW=size_imW,
                E_step=E_step,
            )
            '''
            if net_cfg.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict
        return out_dict

    def gdrn_loss(
        self,
        cfg,
        out_mask,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        out_x,
        out_y,
        out_z,
        gt_xyz,
        gt_xyz_bin,
        out_region,
        gt_region,
        gt_mask_erode=None,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        extents=None,
        # selfocc
        gt_occmask=None,
        gt_Q0=None,
        out_Q0_xy_x=None, out_Q0_xy_y=None, out_Q0_xz_x=None,
        out_Q0_xz_z=None, out_Q0_yz_y=None, out_Q0_yz_z=None,
        roi_extent=None, roi_2d=None, roi_cam=None,
        sizeimH=None,
        sizeimW=None,
        E_step=None,
    ):
        net_cfg = cfg.MODEL.POSE_NET
        g_head_cfg = net_cfg.GEO_HEAD
        s_head_cfg = net_cfg.SELFOCC_HEAD
        pnp_net_cfg = net_cfg.PNP_NET
        loss_cfg = net_cfg.LOSS_CFG

        loss_dict = {}
        if gt_mask_erode is None:
            gt_mask_erode = gt_mask_visib
        gt_masks = {"trunc": gt_mask_trunc, "visib": gt_mask_visib, "obj": gt_mask_obj, "erode": gt_mask_erode}
        ##handle symmetric objects
        if loss_cfg.HANDLE_SYM:
            loss_func = COOR_loss()
            gt_mask_xyz = gt_masks[loss_cfg.XYZ_LOSS_MASK_GT]
            out_p0 = torch.cat([out_x, out_y, out_z], dim=1)
            out_q0 = torch.cat([out_Q0_xy_x, out_Q0_xy_y, out_Q0_xz_x, out_Q0_xz_z, out_Q0_yz_y, out_Q0_yz_z], dim=1)
            loss_dict_coor = loss_func(out_rot, gt_rot,  out_p0, gt_xyz,  out_q0, gt_Q0, gt_occmask, gt_mask_xyz, roi_extent)
            loss_dict.update(loss_dict_coor)
        # Q0 loss ------------------------------------------
        if not s_head_cfg.FREEZE and loss_cfg.Q0_LW > 0.0 and (not loss_cfg.HANDLE_SYM):
            Q0_loss_type = loss_cfg.Q0_LOSS_TYPE
            loss_func = nn.L1Loss(reduction="sum")
            occmask_x = gt_occmask[:, 0, :, :]
            if occmask_x.sum() < s_head_cfg.MIN_Q0_REGION:
                loss_dict["loss_Q0_x"] = 0
            else:
                loss_dict["loss_Q0_x"] = loss_func(out_Q0_xy_x * occmask_x[:, None],
                                                   gt_Q0[:, 0:1] * occmask_x[:, None])\
                                         / occmask_x.sum().float().clamp(min=1.0) + \
                                         loss_func(out_Q0_xy_y * occmask_x[:, None],
                                                   gt_Q0[:, 1:2] * occmask_x[:, None]) \
                                         / occmask_x.sum().float().clamp(min=1.0)

            occmask_y = gt_occmask[:, 1, :, :]
            if occmask_y.sum() < s_head_cfg.MIN_Q0_REGION:
                loss_dict["loss_Q0_y"] = 0
            else:
                loss_dict["loss_Q0_y"] = loss_func(out_Q0_xz_x * occmask_y[:, None],
                                                   gt_Q0[:, 2:3] * occmask_y[:, None]) \
                                         / occmask_y.sum().float().clamp(min=1.0) + \
                                         loss_func(out_Q0_xz_z * occmask_y[:, None],
                                                   gt_Q0[:, 3:4] * occmask_y[:, None]) \
                                         / occmask_y.sum().float().clamp(min=1.0)

            occmask_z = gt_occmask[:, 2, :, :]
            if occmask_z.sum() < s_head_cfg.MIN_Q0_REGION:
                loss_dict["loss_Q0_z"] = 0
            else:
                loss_dict["loss_Q0_z"] = loss_func(out_Q0_yz_y * occmask_z[:, None],
                                                   gt_Q0[:, 4:5] * occmask_z[:, None]) \
                                         / occmask_z.sum().float().clamp(min=1.0) + \
                                         loss_func(out_Q0_yz_z * occmask_z[:, None],
                                                   gt_Q0[:, 5:] * occmask_z[:, None]) \
                                         / occmask_z.sum().float().clamp(min=1.0)
            loss_dict["loss_Q0_x"] *= loss_cfg.Q0_LW
            loss_dict["loss_Q0_y"] *= loss_cfg.Q0_LW
            loss_dict["loss_Q0_z"] *= loss_cfg.Q0_LW
        # cross task loss
        if loss_cfg.CT_LW > 0.0 and E_step > cfg.SOLVER.TOTAL_EPOCHS * cfg.TRAIN.CT_START:
            loss_func = CT_loss(loss_type="L1", loss_weight=loss_cfg.CT_LW)
            out_p0 = torch.cat([out_x, out_y, out_z], dim=1)
            out_q0 = torch.cat([out_Q0_xy_x, out_Q0_xy_y, out_Q0_xz_x, out_Q0_xz_z, out_Q0_yz_y, out_Q0_yz_z], dim=1)
            loss_dict["loss_ct"] = loss_func(pred_rots=out_rot, pred_P0=out_p0, pred_Q0=out_q0,
                                             gt_occmask=gt_occmask, roi_extent=roi_extent, pred_transes=out_trans)
            loss_dict["loss_ct"] *= loss_cfg.CT_LW
        # cross task projection loss
        if loss_cfg.CT_P_LW > 0.0 and E_step > cfg.SOLVER.TOTAL_EPOCHS * cfg.TRAIN.CT_P_START:
            loss_func = CT_loss_projection(loss_type="L1", loss_weight=loss_cfg.CT_P_LW)
            out_p0 = torch.cat([out_x, out_y, out_z], dim=1)
            out_q0 = torch.cat([out_Q0_xy_x, out_Q0_xy_y, out_Q0_xz_x, out_Q0_xz_z, out_Q0_yz_y, out_Q0_yz_z], dim=1)
            loss_dict["loss_ct_pro"] = loss_func(pred_rots=out_rot, pred_P0=out_p0, pred_Q0=out_q0,
                                                 gt_occmask=gt_occmask, roi_extent=roi_extent, pred_transes=out_trans,
                                                 roi_2d=roi_2d, imH=sizeimH, imW=sizeimW, K=roi_cam)
            loss_dict["loss_ct_pro"] *= loss_cfg.CT_P_LW
        # Q0 define loss
        # forward(self, rots, pred_Q0, gt_occmask, roi_extent, transes, roi_2d, imH, imW, K=None):
        if loss_cfg.Q0_DEF_LW > 0.0:
            loss_func = Q_def_loss(loss_type="L1", loss_weight=loss_cfg.CT_P_LW)
            out_q0 = torch.cat([out_Q0_xy_x, out_Q0_xy_y, out_Q0_xz_x, out_Q0_xz_z, out_Q0_yz_y, out_Q0_yz_z], dim=1)
            loss_dict["loss_q_def"] = loss_func(rots=gt_rot, pred_Q0=out_q0, gt_occmask=gt_occmask,
                                                roi_extent=roi_extent, transes=gt_trans, roi_2d=roi_2d,
                                                imH=sizeimH, imW=sizeimW, K=roi_cam)
            loss_dict["loss_q_def"] *= loss_cfg.Q0_DEF_LW
        # xyz loss ----------------------------------
        if not g_head_cfg.FREEZE and (not loss_cfg.HANDLE_SYM):
            xyz_loss_type = loss_cfg.XYZ_LOSS_TYPE
            gt_mask_xyz = gt_masks[loss_cfg.XYZ_LOSS_MASK_GT]
            if xyz_loss_type == "L1":
                loss_func = nn.L1Loss(reduction="sum")
                loss_dict["loss_coor_x"] = loss_func(
                    out_x * gt_mask_xyz[:, None], gt_xyz[:, 0:1] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_y"] = loss_func(
                    out_y * gt_mask_xyz[:, None], gt_xyz[:, 1:2] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_z"] = loss_func(
                    out_z * gt_mask_xyz[:, None], gt_xyz[:, 2:3] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            elif xyz_loss_type == "CE_coor":
                gt_xyz_bin = gt_xyz_bin.long()
                loss_func = CrossEntropyHeatmapLoss(reduction="sum", weight=None)  # g_head_cfg.XYZ_BIN+1
                loss_dict["loss_coor_x"] = loss_func(
                    out_x * gt_mask_xyz[:, None], gt_xyz_bin[:, 0] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_y"] = loss_func(
                    out_y * gt_mask_xyz[:, None], gt_xyz_bin[:, 1] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_z"] = loss_func(
                    out_z * gt_mask_xyz[:, None], gt_xyz_bin[:, 2] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")
            loss_dict["loss_coor_x"] *= loss_cfg.XYZ_LW
            loss_dict["loss_coor_y"] *= loss_cfg.XYZ_LW
            loss_dict["loss_coor_z"] *= loss_cfg.XYZ_LW

        # mask loss ----------------------------------
        if not g_head_cfg.FREEZE:
            mask_loss_type = loss_cfg.MASK_LOSS_TYPE
            gt_mask = gt_masks[loss_cfg.MASK_LOSS_GT]
            if mask_loss_type == "L1":
                loss_dict["loss_mask"] = nn.L1Loss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "BCE":
                loss_dict["loss_mask"] = nn.BCEWithLogitsLoss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "CE":
                loss_dict["loss_mask"] = nn.CrossEntropyLoss(reduction="mean")(out_mask, gt_mask.long())
            else:
                raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
            loss_dict["loss_mask"] *= loss_cfg.MASK_LW

        # roi region loss --------------------
        if not g_head_cfg.FREEZE:
            region_loss_type = loss_cfg.REGION_LOSS_TYPE
            gt_mask_region = gt_masks[loss_cfg.REGION_LOSS_MASK_GT]
            if region_loss_type == "CE":
                gt_region = gt_region.long()
                loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)  # g_head_cfg.XYZ_BIN+1
                loss_dict["loss_region"] = loss_func(
                    out_region * gt_mask_region[:, None], gt_region * gt_mask_region.long()
                ) / gt_mask_region.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown region loss type: {region_loss_type}")
            loss_dict["loss_region"] *= loss_cfg.REGION_LW

        # point matching loss ---------------
        if loss_cfg.PM_LW > 0:
            assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=loss_cfg.PM_LOSS_TYPE,
                beta=loss_cfg.PM_SMOOTH_L1_BETA,
                reduction="mean",
                loss_weight=loss_cfg.PM_LW,
                norm_by_extent=loss_cfg.PM_NORM_BY_EXTENT,
                symmetric=loss_cfg.PM_LOSS_SYM,
                disentangle_t=loss_cfg.PM_DISENTANGLE_T,
                disentangle_z=loss_cfg.PM_DISENTANGLE_Z,
                t_loss_use_points=loss_cfg.PM_T_USE_POINTS,
                r_only=loss_cfg.PM_R_ONLY,
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                extents=extents,
                sym_infos=sym_infos,
            )
            loss_dict.update(loss_pm_dict)

        # rot_loss ----------
        if loss_cfg.ROT_LW > 0:
            if loss_cfg.ROT_LOSS_TYPE == "angular":
                loss_dict["loss_rot"] = angular_distance(out_rot, gt_rot)
            elif loss_cfg.ROT_LOSS_TYPE == "L2":
                loss_dict["loss_rot"] = rot_l2_loss(out_rot, gt_rot)
            else:
                raise ValueError(f"Unknown rot loss type: {loss_cfg.ROT_LOSS_TYPE}")
            loss_dict["loss_rot"] *= loss_cfg.ROT_LW

        # centroid loss -------------
        if loss_cfg.CENTROID_LW > 0:
            assert (
                pnp_net_cfg.TRANS_TYPE == "centroid_z"
            ), "centroid loss is only valid for predicting centroid2d_rel_delta"

            if loss_cfg.CENTROID_LOSS_TYPE == "L1":
                loss_dict["loss_centroid"] = nn.L1Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif loss_cfg.CENTROID_LOSS_TYPE == "L2":
                loss_dict["loss_centroid"] = L2Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif loss_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_centroid"] = nn.MSELoss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            else:
                raise ValueError(f"Unknown centroid loss type: {loss_cfg.CENTROID_LOSS_TYPE}")
            loss_dict["loss_centroid"] *= loss_cfg.CENTROID_LW

        # z loss ------------------
        if loss_cfg.Z_LW > 0:
            z_type = pnp_net_cfg.Z_TYPE
            if z_type == "REL":
                gt_z = gt_trans_ratio[:, 2]
            elif z_type == "ABS":
                gt_z = gt_trans[:, 2]
            else:
                raise NotImplementedError

            z_loss_type = loss_cfg.Z_LOSS_TYPE
            if z_loss_type == "L1":
                loss_dict["loss_z"] = nn.L1Loss(reduction="mean")(out_trans_z, gt_z)
            elif z_loss_type == "L2":
                loss_dict["loss_z"] = L2Loss(reduction="mean")(out_trans_z, gt_z)
            elif z_loss_type == "MSE":
                loss_dict["loss_z"] = nn.MSELoss(reduction="mean")(out_trans_z, gt_z)
            else:
                raise ValueError(f"Unknown z loss type: {z_loss_type}")
            loss_dict["loss_z"] *= loss_cfg.Z_LW

        # trans loss ------------------
        if loss_cfg.TRANS_LW > 0:
            if loss_cfg.TRANS_LOSS_DISENTANGLE:
                # NOTE: disentangle xy/z
                if loss_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_xy"] = nn.L1Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_xy"] = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_xy"] = nn.MSELoss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_xy"] *= loss_cfg.TRANS_LW
                loss_dict["loss_trans_z"] *= loss_cfg.TRANS_LW
            else:
                if loss_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_LPnP"] = nn.L1Loss(reduction="mean")(out_trans, gt_trans)
                elif loss_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_LPnP"] = L2Loss(reduction="mean")(out_trans, gt_trans)

                elif loss_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_LPnP"] = nn.MSELoss(reduction="mean")(out_trans, gt_trans)
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_LPnP"] *= loss_cfg.TRANS_LW

        # bind loss (R^T@t)
        if loss_cfg.get("BIND_LW", 0.0) > 0.0:
            pred_bind = torch.bmm(out_rot.permute(0, 2, 1), out_trans.view(-1, 3, 1)).view(-1, 3)
            gt_bind = torch.bmm(gt_rot.permute(0, 2, 1), gt_trans.view(-1, 3, 1)).view(-1, 3)
            if loss_cfg.BIND_LOSS_TYPE == "L1":
                loss_dict["loss_bind"] = nn.L1Loss(reduction="mean")(pred_bind, gt_bind)
            elif loss_cfg.BIND_LOSS_TYPE == "L2":
                loss_dict["loss_bind"] = L2Loss(reduction="mean")(pred_bind, gt_bind)
            elif loss_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_bind"] = nn.MSELoss(reduction="mean")(pred_bind, gt_bind)
            else:
                raise ValueError(f"Unknown bind loss (R^T@t) type: {loss_cfg.BIND_LOSS_TYPE}")
            loss_dict["loss_bind"] *= loss_cfg.BIND_LW

        if net_cfg.USE_MTL:
            for _k in loss_dict:
                _name = _k.replace("loss_", "log_var_")
                cur_log_var = getattr(self, _name)
                loss_dict[_k] = loss_dict[_k] * torch.exp(-cur_log_var) + torch.log(1 + torch.exp(cur_log_var))
        return loss_dict


def build_model_optimizer(cfg, is_test=False):
    net_cfg = cfg.MODEL.POSE_NET
    backbone_cfg = net_cfg.BACKBONE

    params_lr_list = []
    # backbone
    backbone_type = backbone_cfg.INIT_CFG.pop("type")
    init_backbone_args = copy.deepcopy(backbone_cfg.INIT_CFG)
    if "timm/" in backbone_type or "tv/" in backbone_type:
        init_backbone_args["model_name"] = backbone_type.split("/")[-1]

    backbone = BACKBONES[backbone_type](**init_backbone_args)
    if backbone_cfg.FREEZE:
        for param in backbone.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, backbone.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )

    # neck --------------------------------
    neck, neck_params = get_neck(cfg)
    params_lr_list.extend(neck_params)

    # geo head -----------------------------------------------------
    geo_head, geo_head_params = get_geo_head(cfg)
    params_lr_list.extend(geo_head_params)

    # selfocc head
    selfocc_head, selfocc_head_params = get_selfocc_head(cfg)
    params_lr_list.extend(selfocc_head_params)
    # pnp net -----------------------------------------------
    pnp_net, pnp_net_params = get_pnp_net(cfg)
    params_lr_list.extend(pnp_net_params)

    # build model
    model = GDRN(cfg, backbone, neck=neck, geo_head_net=geo_head, selfocc_head_net=selfocc_head, pnp_net=pnp_net)
    if net_cfg.USE_MTL:
        params_lr_list.append(
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [_param for _name, _param in model.named_parameters() if "log_var" in _name],
                ),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )

    # get optimizer
    if is_test:
        optimizer = None
    else:
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        backbone_pretrained = backbone_cfg.get("PRETRAINED", "")
        if backbone_pretrained == "":
            logger.warning("Randomly initialize weights for backbone!")
        elif backbone_pretrained in ["timm", "internal"]:
            # skip if it has already been initialized by pretrained=True
            logger.info("Check if the backbone has been initialized with its own method!")
        else:
            # initialize backbone with official weights
            tic = time.time()
            logger.info(f"load backbone weights from: {backbone_pretrained}")
            load_checkpoint(model.backbone, backbone_pretrained, strict=False, logger=logger)
            logger.info(f"load backbone weights took: {time.time() - tic}s")

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer
