import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


from .losses_depth import *
from .occlusion_loss import *
from .losses_depthob import *

PI = 3.1416


class DepthOBLoss(nn.Module):
    def __init__(self, args):
        super(DepthOBLoss, self).__init__()
        self.tasks = ["depth", "ob"]
        self.auxilary_tasks = ["depth", "ob"]


        print('=> Init criterions')

        if args.loss_type == "silog":
            self.depth_criterion = silog_loss(variance_focus=args.variance_focus)
        elif args.loss_type == "berhu":
            self.depth_criterion = LainaBerHuLoss(use_logs=True)
        elif args.loss_type == "huber":
            self.depth_criterion = HuberLoss()
        elif args.loss_type == "l1":
            self.depth_criterion = MTl1loss()     # L1Loss(loss_gamma=1.0)
        elif args.loss_type == "l2":
            self.depth_criterion = L2Loss(loss_gamma=1.0)
        elif args.loss_type == "none":
            self.depth_criterion = None     
            args.use_geo_consensus = False     
        else:
            raise NotImplementedError
        self.depth_loss = args.loss_type

        if args.ob_loss_type == "cce":
            boundary_weights = args.boundary_weights.split(',')
            boundary_weights = list(map(float, boundary_weights))
            print(f' Occlusion side boundary_weights: {boundary_weights}')
            print(f' Occlusion boundary_lambda: {args.boundary_lambda}')
            print(f' OB loss weight: {args.ob_loss_weight}')

            if args.model_name not in ["invpt", "mlore", "taskprompter"]:
                self.ob_criterion = CCELoss(b_lambda=args.boundary_lambda,boundary_weights=boundary_weights)
            else:
                print()
                print("For NYUD-MT padded GT, use masked CCE loss")
                self.ob_criterion = Mask_CCELoss(b_lambda=args.boundary_lambda,boundary_weights=boundary_weights)
                print(self.ob_criterion)
                print()

        elif args.ob_loss_type == "al":
            self.ob_criterion = AttentionLoss()
        elif args.ob_loss_type == "fl":
            self.ob_criterion = FocalLossV2()
        # trian depth with normal
        elif args.ob_loss_type == "normal" and args.hypersim_b_type == "normal":
            boundary_weights = args.boundary_weights.split(',')
            boundary_weights = list(map(float, boundary_weights))
            self.ob_criterion = NormalsLoss(normalize=True, size_average=True, norm=1, boundary_weights=boundary_weights)
        elif args.ob_loss_type == "sem_seg" and args.hypersim_b_type == "sem_seg":
            self.ob_criterion = SoftMaxwithLoss()
        elif args.ob_loss_type == "none":
            self.ob_criterion = None
        else:
            raise NotImplementedError
        
        self.ob_loss_type = args.ob_loss_type
        
        
        # only for final depth ?
        self.use_geo_consensus = args.use_geo_consensus
        self.geo_use_gt = True
        if self.use_geo_consensus:
            self.depth_bound_consensus_loss = OBDCL(args.dataset, nyudmt_ob=args.hypersim_add_contour)
        
        self.dataset = args.dataset

        self.final_ob_loss_weights = args.ob_loss_weight
        self.final_depth_loss_weights = args.depth_loss_weight
        self.initial_loss_weights = args.init_depth_loss_weight

        self.use_step_OBDCL = args.use_step_OBDCL

        if args.dataset == "nyudmt":
            self.step_geol = 9000
        elif args.dataset == "hypersim":
            self.step_geol = 20000  # 25000
        else:
            self.step_geol = 0
        if args.is_two_stage:
            self.step_geol = 0
        print("step Geo Loss, start step: ", self.step_geol)


    def forward(self, out, depth_gt, ob_gt, current_step):

        if self.dataset == 'synocc':
            mask = (depth_gt < 10.0) & (depth_gt > 0.0)
        elif self.dataset == 'hypersim':
            mask = (~torch.isnan(depth_gt)) & (depth_gt > 0.0)
        elif self.dataset == 'nyudmt':
            mask = (depth_gt != 255) # fill 0 but ignore 255 
        elif self.dataset == 'nyudmt_reverse':
            mask = (depth_gt != 0) 
            # Define crop area mask (e.g., standard Eigen crop: 45:471, 41:601)
            crop_mask = torch.zeros_like(depth_gt, dtype=torch.bool)
            crop_mask[:, :, 45:471, 41:601] = True 
            mask = mask & crop_mask
            # num_valid_pixels = mask.sum().item()

        if self.depth_loss != "l1":
            mask = mask.to(torch.bool)

        if self.depth_criterion:
            depth_loss = self.final_depth_loss_weights * self.depth_criterion.forward(out[0], depth_gt, mask) #  mask.to(torch.bool))
            # if torch.isnan(depth_loss):
            #     depth_loss = 0.0
        else:
            depth_loss = torch.tensor(0.0, device=out[0].device)


        if self.ob_criterion:
            if self.ob_loss_type == "sem_seg":
                ob_loss = self.final_ob_loss_weights * self.ob_criterion(out[1][-1], ob_gt)
            else:
                ob_loss = self.final_ob_loss_weights * self.ob_criterion(out[1], ob_gt)
        else:
            ob_loss = torch.tensor(0.0, device=out[0].device)
        
        if len(out) == 3:
            for i in range(len(out[-1])):
                aux_depth_loss = self.initial_loss_weights * self.depth_criterion.forward(out[-1][i], depth_gt, mask) # mask.to(torch.bool))
                if not torch.isnan(aux_depth_loss):
                    depth_loss += aux_depth_loss

        # OBDCL
        if self.use_geo_consensus and self.use_step_OBDCL and current_step >= self.step_geol:
            # if self.geo_use_gt:
                # use bound pred or bound gt ?
            geo_loss = self.depth_bound_consensus_loss(out[0], ob_gt)
            # else:
            #     geo_loss = self.depth_bound_consensus_loss(out[0], out[1])
            
        else:
            geo_loss = torch.tensor(0.0, device=out[0].device)

        return depth_loss, ob_loss, geo_loss





class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        # print(len(bottom), type(bottom))
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(nn.Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """
    def __init__(self, size_average=True, normalize=False, norm=1, boundary_weights=[]):
        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError
        
        self.boundary_weights = 1.0  # boundary_weights

    def forward(self, out, label, ignore_label=255):
        # out = out[-1]
        assert not label.requires_grad
        # assert len(self.boundary_weights) == len(out)
        mask = (~torch.isnan(label))
        n_valid = torch.sum(mask).item()

        total_loss = 0.0

        # for (item_weights, item_out) in enumerate(self.boundary_weights, out):
        # for i in range(len(out)):

        item_weights = 1.0  # self.boundary_weights[i]
        item_out = out[-1]
        if self.normalize is not None:
            # print(len(out), type(out))
            out_norm = self.normalize(item_out)
            loss = self.loss_func(torch.masked_select(out_norm, mask), torch.masked_select(label, mask), reduction='sum')
        else:
            loss = self.loss_func(torch.masked_select(item_out, mask), torch.masked_select(label, mask), reduction='sum')

        if self.size_average:
            if ignore_label:
                ret_loss = item_weights * torch.div(loss, max(n_valid, 1e-6))
                total_loss += ret_loss
                return ret_loss
                # continue
            else:
                ret_loss = item_weights *  torch.div(loss, float(np.prod(label.size())))
                total_loss += ret_loss
                return ret_loss
                # continue

        return total_loss


class SoftMaxwithLoss(nn.Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=255)

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        # print(label.shape, out.shape)
        loss = self.criterion(self.softmax(out), label)

        return loss