# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
from torch import log as thLog
from torch.autograd import Variable
from torch import Tensor, mul, dot, ones


import torch
import torch.nn as nn
import torch.nn.functional as F


class OBDCL(nn.Module):
    def __init__(self, train_dataset, boundary_weight=0.1, smoothing_weight=0.1, neighbor_weight=0.1, clamp_value=1e-7, neighbor_size=3,nyudmt_ob=False):

        super().__init__()
        self.boundary_weight =  0.1  

        print("OBDCL weight: ", self.boundary_weight)
        self.smoothing_weight = smoothing_weight
        self.neighbor_weight = neighbor_weight
        self.clamp_value = clamp_value
        self.neighbor_size = neighbor_size
        self.is_smoothing = False 

        self.lambda_ob = 1.0

        if train_dataset in ["synocc", "nyudmt", "nyudmt_reverse"]: 
            self.only_rl=True
            self.full_diff=False
            self.is_smoothing = True
        else:
            self.only_rl = True
            self.full_diff = False
            self.is_smoothing = False

        # Neighbor dilation kernel (square window)
        self.neighbor_kernel = nn.Parameter(
            torch.ones(1, 1, neighbor_size, neighbor_size), requires_grad=False
        )

    def depth_discontinuity_constraint_v1(self, depth, ob, lambda_ob=1.0, only_rl=True, full_diff=False):
        """Encourage significant depth differences across OB regions."""
        # Shift depth values for all four neighbors
        depth_up = F.pad(depth[:, :, :-1, :], (0, 0, 1, 0))  # Shift depth upward
        depth_down = F.pad(depth[:, :, 1:, :], (0, 0, 0, 1))  # Shift depth downward
        depth_left = F.pad(depth[:, :, :, :-1], (1, 0, 0, 0))  # Shift depth to the left
        depth_right = F.pad(depth[:, :, :, 1:], (0, 1, 0, 0))  # Shift depth to the right

        # Compute depth differences for all four neighbors
        depth_diff_down = torch.abs(depth_up - depth_down)
        depth_diff_right = torch.abs(depth_left - depth_right)

        depth_diff = torch.min(depth_diff_down, depth_diff_right)
        # if full_diff:
        #     depth_diff = depth_diff_down + depth_diff_right
        # elif only_rl:
        #     depth_diff = depth_diff_right
        # else:
        #     # not test
        #     depth_diff = depth_diff_down

        # Apply penalty: encourage significant depth differences at OB regions
        ob_penalty = torch.mean(ob * (1.0 - depth_diff.clamp(min=0, max=1)))
        
        return lambda_ob * ob_penalty

    def depth_smoothness_constraint(self, depth, ob, lambda_smooth=1.0, radius=1):
        """Encourage smooth depth transitions in non-OB regions."""
        # Shift depth values
        depth_up = F.pad(depth[:, :, :-radius, :], (0, 0, radius, 0))
        depth_down = F.pad(depth[:, :, radius:, :], (0, 0, 0, radius))
        depth_left = F.pad(depth[:, :, :, :-radius], (radius, 0, 0, 0))
        depth_right = F.pad(depth[:, :, :, radius:], (0, radius, 0, 0))
        
        # Compute smoothness loss for non-OB regions
        smoothness_loss = (
            torch.mean((1 - ob) * torch.abs(depth - depth_up)) +
            torch.mean((1 - ob) * torch.abs(depth - depth_down)) +
            torch.mean((1 - ob) * torch.abs(depth - depth_left)) +
            torch.mean((1 - ob) * torch.abs(depth - depth_right))
        )
        
        return lambda_smooth * smoothness_loss

    def compute_neighborhood_mask(self, boundary_mask):
        """Dilate boundary mask to include neighbors."""
        neighbor_kernel = self.neighbor_kernel.to(boundary_mask.device)
        dilated_mask = F.conv2d(boundary_mask, neighbor_kernel, padding=self.neighbor_size // 2)
        neighborhood_mask = (dilated_mask > 0).float()  # Convert to binary mask
        return neighborhood_mask

    def forward(self, depth_pred, ob_gt):

        boundary_mask = (ob_gt > 0).float()

        # neighborhood_mask = self.compute_neighborhood_mask(boundary_mask)
        # ob_gt = (boundary_mask + neighborhood_mask).clamp(max=1)

        ob_near_penalty = self.boundary_weight * self.depth_discontinuity_constraint_v1(depth_pred, boundary_mask, 
                                                                                        lambda_ob=1.0, only_rl=self.only_rl, full_diff=self.full_diff) 
        if self.is_smoothing:
            ob_far_penalty = self.smoothing_weight * self.depth_smoothness_constraint(depth_pred, boundary_mask, lambda_smooth=1.0)
        else:
            ob_far_penalty = 0.0

        geo_loss = ob_near_penalty + ob_far_penalty
        
        return geo_loss


class DepthGradBoundaryLoss(nn.Module):
    def __init__(self, kernel_size=3, clamp_value=1e-7, size_average=False):
        super().__init__()

        self.size_average = size_average
        self.kernel_size = kernel_size
        self.clamp_value = clamp_value

    def forward(self, depth, boundary, mask=None):
        repeat_channels = depth.shape[1]

        sobel_x = torch.Tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=depth.device).view(1, 1, 3, 3)

        sobel_x = torch.autograd.Variable(sobel_x)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32, device=depth.device).view(1, 1, 3, 3)

        sobel_y = torch.autograd.Variable(sobel_y)

        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=depth.device).view(1, 1, 3, 3)
        lap = torch.autograd.Variable(lap)

        if repeat_channels != 1:
            sobel_x = sobel_x.repeat(1, repeat_channels, 1, 1)
            sobel_y = sobel_y.repeat(1, repeat_channels, 1, 1)
            lap = lap.repeat(1, repeat_channels, 1, 1)

        # Compute Laplacian of depth
        lap_depth = F.conv2d(depth, (1 / 8.0) * lap, padding=1)

        # Compute Sobel gradients
        gx = F.conv2d(depth, (1.0 / 8.0) * sobel_x, padding=1)
        gy = F.conv2d(depth, (1.0 / 8.0) * sobel_y, padding=1)
        g_depth = torch.pow(gx, 2) + torch.pow(gy, 2)

        # Clamp boundary ground truth values for stability
        # boundary_gt = boundary_gt.clamp(min=self.clamp_value, max=1.0)

        # Compute loss
        term1 = torch.abs(boundary_gt * g_depth * lap_depth)  # Weight gradients by boundary ground truth
        term2 = 0.0001 * torch.abs((1 - boundary_gt) * torch.exp(-lap_depth))  # Penalize non-boundary regions
        loss = term1 + term2

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1e-7)
        else:
            return loss.mean() if self.size_average else loss.sum() / depth.numel()


class GradientAlignmentLoss(torch.nn.Module):
    def __init__(self):
        super(GradientAlignmentLoss, self).__init__()
        self.sobel_x = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, depth_pred, boundary_gt):
        device = depth_pred.device
        self.sobel_x, self.sobel_y = self.sobel_x.to(device), self.sobel_y.to(device)

        # Compute depth gradients
        gx = F.conv2d(depth_pred, self.sobel_x, padding=1)
        gy = F.conv2d(depth_pred, self.sobel_y, padding=1)
        g_depth = torch.sqrt(gx**2 + gy**2 + 1e-8)

        # Loss based on the similarity between depth gradient and boundary GT
        loss = torch.abs(boundary_gt - g_depth)
        loss = torch.mean((boundary_gt - g_depth)**2)
        # F.binary_cross_entropy_with_logits(depth_pred, boundary_gt)
        
        return loss.mean()


def edge_aware_smoothness_loss(depth_pred, boundary_gt, image):
    """
    Edge-aware smoothness loss to encourage depth smoothness in non-boundary regions.

    Args:
        depth_pred (torch.Tensor): Predicted depth map (B, 1, H, W)
        boundary_gt (torch.Tensor): Ground-truth occlusion boundary (B, 1, H, W)
        image (torch.Tensor): Input image for edge-awareness (B, 3, H, W)

    Returns:
        torch.Tensor: Edge-aware smoothness loss
    """
    # Compute image gradients
    gx_img = F.conv2d(image, torch.tensor([[1, 0, -1],
                                           [2, 0, -2],
                                           [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device), padding=1)
    gy_img = F.conv2d(image, torch.tensor([[1, 2, 1],
                                           [0, 0, 0],
                                           [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device), padding=1)
    img_grad = torch.sqrt(gx_img**2 + gy_img**2 + 1e-8)

    # Compute depth gradients
    gx_depth = F.conv2d(depth_pred, torch.tensor([[1, 0, -1],
                                                  [2, 0, -2],
                                                  [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(depth_pred.device), padding=1)
    gy_depth = F.conv2d(depth_pred, torch.tensor([[1, 2, 1],
                                                  [0, 0, 0],
                                                  [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(depth_pred.device), padding=1)
    depth_grad = torch.sqrt(gx_depth**2 + gy_depth**2 + 1e-8)

    # Boundary weight
    boundary_weight = 1 - boundary_gt

    # Edge-aware smoothness
    weight = torch.exp(-img_grad)
    loss = torch.mean(weight * boundary_weight * depth_grad)

    return loss

def gradient_difference_loss(depth_pred, boundary_gt, sobel_x, sobel_y):
    """
    Gradient difference loss to match depth gradients to boundary gradients.

    Args:
        depth_pred (torch.Tensor): Predicted depth map (B, 1, H, W)
        boundary_gt (torch.Tensor): Ground-truth occlusion boundary (B, 1, H, W)
        sobel_x (torch.Tensor): Sobel filter for x-direction gradients
        sobel_y (torch.Tensor): Sobel filter for y-direction gradients

    Returns:
        torch.Tensor: Gradient difference loss
    """
    # Compute depth gradients
    gx_depth = F.conv2d(depth_pred, sobel_x.to(depth_pred.device), padding=1)
    gy_depth = F.conv2d(depth_pred, sobel_y.to(depth_pred.device), padding=1)

    # Compute boundary gradients
    gx_boundary = F.conv2d(boundary_gt, sobel_x.to(boundary_gt.device), padding=1)
    gy_boundary = F.conv2d(boundary_gt, sobel_y.to(boundary_gt.device), padding=1)

    # L1 difference between gradients
    loss = torch.mean(torch.abs(gx_depth - gx_boundary) + torch.abs(gy_depth - gy_boundary))
    return loss


def combined_loss(depth_pred, boundary_gt, image, sobel_x, sobel_y):
    """
    Combined loss: smoothness + boundary alignment + gradient difference.

    Args:
        depth_pred (torch.Tensor): Predicted depth map
        boundary_gt (torch.Tensor): Ground-truth occlusion boundary
        image (torch.Tensor): Input image
        sobel_x, sobel_y (torch.Tensor): Sobel filters

    Returns:
        torch.Tensor: Total loss
    """
    smooth_loss = edge_aware_smoothness_loss(depth_pred, boundary_gt, image)
    boundary_loss = boundary_alignment_loss(depth_pred, boundary_gt, sobel_x, sobel_y)
    grad_diff_loss = gradient_difference_loss(depth_pred, boundary_gt, sobel_x, sobel_y)

    # Combine with weights
    total_loss = 0.5 * smooth_loss + 0.3 * boundary_loss + 0.2 * grad_diff_loss
    return total_loss


class BoundaryConstrainedLoss(torch.nn.Module):
    def __init__(self):
        super(BoundaryConstrainedLoss, self).__init__()
        self.lap = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, depth_pred, boundary_gt):
        device = depth_pred.device
        self.lap = self.lap.to(device)

        # Apply Laplacian filter
        lap_depth = F.conv2d(depth_pred, self.lap, padding=1)

        # Compute loss based on Laplacian and boundary GT
        loss = torch.abs(lap_depth - boundary_gt)
        #F.binary_cross_entropy_with_logits(depth_pred, boundary_gt)
        
        return loss.mean()


class SmoothnessBoundaryLoss(torch.nn.Module):
    def __init__(self, smoothness_weight=0.1):
        super(SmoothnessBoundaryLoss, self).__init__()
        self.smoothness_weight = smoothness_weight

    def forward(self, depth_pred, boundary_gt):
        # Compute gradients of depth
        gx_depth = torch.abs(depth_pred[:, :, :, 1:] - depth_pred[:, :, :, :-1])
        gy_depth = torch.abs(depth_pred[:, :, 1:, :] - depth_pred[:, :, :-1, :])

        # Smoothness loss
        smoothness_loss = gx_depth.mean() + gy_depth.mean()

        # Boundary loss (binary cross-entropy)
        boundary_loss = F.binary_cross_entropy_with_logits(depth_pred, boundary_gt)

        return self.smoothness_weight * smoothness_loss + boundary_loss

def smoothness_loss(depth_pred, boundary_gt, sobel_x, sobel_y):
    """
    Smoothness loss: regularizes depth in non-boundary regions.

    Args:
        depth_pred (torch.Tensor): Predicted depth map (B, 1, H, W)
        boundary_gt (torch.Tensor): Ground-truth occlusion boundary (B, 1, H, W)
        sobel_x (torch.Tensor): Pre-defined Sobel filter for x-direction gradients
        sobel_y (torch.Tensor): Pre-defined Sobel filter for y-direction gradients

    Returns:
        torch.Tensor: Smoothness loss value
    """
    # Compute depth gradients using Sobel filters
    gx = F.conv2d(depth_pred, sobel_x.to(depth_pred.device), padding=1)
    gy = F.conv2d(depth_pred, sobel_y.to(depth_pred.device), padding=1)
    g_depth = torch.sqrt(gx**2 + gy**2 + 1e-8)

    # Invert boundary ground truth (non-boundary regions)
    non_boundary = 1 - boundary_gt

    # Apply the non-boundary mask to the gradient magnitude
    loss = torch.mean(non_boundary * g_depth)
    return loss


class ChamferLikeDistanceLoss(nn.Module):
    def __init__(self):
        super(ChamferLikeDistanceLoss, self).__init__()
        self.sobel_x = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def compute_depth_gradients(self, depth_pred):
        # Compute gradients of the depth prediction
        gx = F.conv2d(depth_pred, self.sobel_x.to(depth_pred.device), padding=1)
        gy = F.conv2d(depth_pred, self.sobel_y.to(depth_pred.device), padding=1)
        g_depth = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return g_depth

    def forward(self, depth_pred, boundary_gt):
        """
        depth_pred: Predicted depth map (B, 1, H, W)
        boundary_gt: Occlusion boundary ground truth (B, 1, H, W)
        """
        # Compute the magnitude of gradients for depth prediction
        g_depth = self.compute_depth_gradients(depth_pred)

        # Flatten the tensors for distance calculation
        g_depth_flat = g_depth.view(g_depth.size(0), -1)  # (B, N)
        boundary_gt_flat = boundary_gt.view(boundary_gt.size(0), -1)  # (B, N)

        # Compute distances (symmetric Chamfer-like distance)
        # Closest depth gradient to each boundary pixel
        dist1 = torch.min(torch.abs(g_depth_flat.unsqueeze(2) - boundary_gt_flat.unsqueeze(1)), dim=2).values
        # Closest boundary pixel to each depth gradient
        dist2 = torch.min(torch.abs(boundary_gt_flat.unsqueeze(2) - g_depth_flat.unsqueeze(1)), dim=2).values

        # Final loss is the mean of the distances
        loss = dist1.mean() + dist2.mean()
        return loss



