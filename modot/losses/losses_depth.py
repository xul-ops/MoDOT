# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
from torch import log as thLog
from torch.autograd import Variable
from torch import Tensor, mul, dot, ones



class BinsChamferLoss(nn.Module):
    """BinsChamferLoss used in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_. 
    
        Waiting for re-writing
        
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 loss_weight=1.0):
        super(BinsChamferLoss, self).__init__()
        self.loss_weight = loss_weight

    def bins_chamfer_loss(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

    def forward(self,
                input,
                target):
        """Forward function."""
        
        chamfer_loss = self.bins_chamfer_loss(input, target)
        chamfer_loss = self.loss_weight * chamfer_loss
        return chamfer_loss
        


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus
        self.eps = torch.Tensor([1e-13]).cuda()

    def forward(self, depth_est, depth_gt, mask):


        mask_t = mask & (depth_est > 0)
        mask_t = mask_t.to(torch.bool)

        # print("Mask shape:", mask.shape)
        # print("Number of valid elements in mask:", mask.sum().item())
        # print("Depth estimate has NaN:", torch.isnan(depth_est).any().item())
        # print("Depth ground truth has NaN:", torch.isnan(depth_gt).any().item())
        # print("Depth estimate has negative values:", (depth_est < 0).any().item())
        # print("Depth ground truth has negative values:", (depth_gt < 0).any().item())

        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        diff = (d ** 2).mean() - self.variance_focus * (d.mean() ** 2)
        diff = torch.where(diff < self.eps, self.eps, diff)
        # diff = torch.sqrt(diff)
        return torch.sqrt(diff) # * 10.0


class SigLoss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001 # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth
        

class LainaBerHuLoss(nn.Module):
    # Based on Laina et al.

    def __init__(self, size_average=True, use_logs=True, clamp_val=1e-9):
        super(LainaBerHuLoss, self).__init__()
        self.size_average = size_average
        self.use_log = use_logs
        self.clamp_val = clamp_val

    def forward(self, input, target, mask):
        if self.use_log:
            n = thLog(input.clamp(min=self.clamp_val)) - thLog(target.clamp(min=self.clamp_val))
        else:
            n = input - target


        mask_t = mask & (input > 0)
        mask = mask_t.to(torch.bool)

        n = torch.abs(n)
        n = mul(n, mask)

        n = n.squeeze(1)
        c = 0.2 * n.max()
        cond = n < c
        loss = torch.where(cond, n, (n ** 2 + c ** 2) / (2 * c + 1e-9))

        loss = loss.sum()

        if self.size_average:
            return loss / mask.sum()

        return loss

class HuberLoss(nn.Module):
    def __init__(self, size_average=True, use_logs=True, sigma=1):
        super(HuberLoss, self).__init__()
        self.size_average = size_average
        self.sigma = sigma

    def forward(self, input, target, mask=None):

        mask_t = mask & (input > 0)
        mask = mask_t.to(torch.bool)

        n = torch.abs(input - target)
        if mask is not None:
            n = mul(n, mask)

        cond = n < 1 / (self.sigma ** 2)
        loss = torch.where(cond, 0.5 * (self.sigma * n) ** 2, n - 0.5 / (self.sigma ** 2))
        if self.size_average:
            if mask is not None:
                return loss.sum() / mask.sum()
            else:
                return loss.mean()
        return loss.sum()
        

def normals_loss(input, target, mask=None):
    if input is None or target is None:
        return 0
    else:
        prod = mul(input, target)

        if mask is not None:
            n = mask.sum().float()
            prod = mul(prod, mask)
        else:
            n = target.numel().float()

        prod = 1.0 - (1.0 / n) * prod.sum()
        prod = prod.clamp(min=0)

        return prod


class SpatialGradientsLoss(nn.Module):
    def __init__(self, kernel_size=3, use_logs=True, clamp_value=1e-7, size_average=False,
                 smooth_error=True,
                 gradient_loss_on=True):
        super(SpatialGradientsLoss, self).__init__()

        self.size_average = size_average
        self.kernel_size = kernel_size
        self.clamp_value = clamp_value
        self.use_logs = use_logs
        self.smooth_error = smooth_error
        self.gradient_loss_on = gradient_loss_on

        if gradient_loss_on:
            self.masked_huber_loss = HuberLoss(sigma=3)

    def forward(self, input, target, mask=None):

        repeat_channels = target.shape[1]

        sobel_x = torch.Tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])

        sobel_x = sobel_x.view((1, 1, 3, 3))
        sobel_x = torch.autograd.Variable(sobel_x.cuda())

        sobel_y = torch.Tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])

        sobel_y = sobel_y.view((1, 1, 3, 3))
        sobel_y = torch.autograd.Variable(sobel_y.cuda())
        if repeat_channels != 1:
            sobel_x = sobel_x.repeat(1, repeat_channels, 1, 1)
            sobel_y = sobel_y.repeat(1, repeat_channels, 1, 1)

        smooth_loss = 0
        grad_loss = 0

        if self.smooth_error:
            diff = thLog(input.clamp(min=self.clamp_value)) - thLog(target.clamp(min=self.clamp_value))

            gx_diff = F.conv2d(diff, (1.0 / 8.0) * sobel_x, padding=1)
            gy_diff = F.conv2d(diff, (1.0 / 8.0) * sobel_y, padding=1)

            gradients_diff = torch.pow(gx_diff, 2) + torch.pow(gy_diff, 2)

            if mask is None:
                smooth_loss = gradients_diff.sum()
                if self.size_average:
                    smooth_loss = smooth_loss * (1.0 / gradients_diff.numel())
            else:
                gradients_diff = mul(gradients_diff, mask.repeat(1, 3, 1, 1))
                smooth_loss = gradients_diff.sum()
                if self.size_average:
                    smooth_loss = smooth_loss * (1.0 / mask.sum())

        if self.gradient_loss_on:

            input = thLog(input.clamp(min=self.clamp_value))
            target = thLog(target.clamp(min=self.clamp_value))

            gx_input = F.conv2d(input, (1.0 / 8.0) * sobel_x, padding=1)
            gy_input = F.conv2d(input, (1.0 / 8.0) * sobel_y, padding=1)

            gx_target = F.conv2d(target, (1.0 / 8.0) * sobel_x, padding=1)
            gy_target = F.conv2d(target, (1.0 / 8.0) * sobel_y, padding=1)

            gradients_input = torch.pow(gx_input, 2) + torch.pow(gy_input, 2)
            gradients_target = torch.pow(gx_target, 2) + torch.pow(gy_target, 2)

            grad_loss = self.masked_huber_loss(gradients_input, gradients_target, mask)

        return smooth_loss + grad_loss


        
class L2loss(nn.Module):
    def __init__(self, loss_gamma=1.0):
        super().__init__()
        self.gamma = loss_gamma
        self.l2_loss = nn.MSELoss()

    def forward(self, pred_list, gt_depth, gt_depth_mask):
        n_predictions = len(pred_list)
        loss = 0.0

        for i in range(n_predictions):
            pred = pred_list[i]
            i_weight = self.gamma ** (n_predictions - i - 1)
            gt_depth = gt_depth[gt_depth_mask]
            pred = pred[gt_depth_mask]
            loss = loss + i_weight * self.l2_loss(pred, gt_depth)

        return loss

class MTl1loss(nn.Module):

    def __init__(self,loss="l1"):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, out, label, mask=None):
        if mask is not None:
            # print(mask)
            depth_loss = self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))
        else:
            depth_loss = self.loss(out, label)
        return depth_loss
        
class L1loss(nn.Module):
    def __init__(self, loss_gamma=1.0):
        super().__init__()
        self.gamma = loss_gamma

    def forward(self, pred_list, gt_depth, gt_depth_mask):
        n_predictions = len(pred_list)
        loss = 0.0

        for i in range(n_predictions):
            pred = pred_list[i]
            i_weight = self.gamma ** (n_predictions - i - 1)
            loss = loss + i_weight * self.l1_loss(pred, gt_depth, gt_depth_mask)

        return loss

    def l1_loss(self, out, gt_depth, gt_depth_mask):
        gt_depth = gt_depth[gt_depth_mask]
        pred_depth = out[gt_depth_mask]
        l1 = torch.abs(pred_depth - gt_depth)
        return torch.mean(l1)
