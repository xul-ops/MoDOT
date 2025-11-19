import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

PI = 3.1416


class BaseOriLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.type_loss = ''

    def forward(self, ori_x, ori_y, edge_y):
        pass

    def getOri(self, ori_x):
        return ori_x
        pass

    def getOriXNnm(self):
        return 1


class OORLoss(BaseOriLoss):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ori_x, ori_y, b_y):
        # b_mask = b_y == 1
        # ori_y = ori_y[b_mask]
        ori_x_sin, ori_x_cos = ori_x[:, 0], ori_x[:, 1]

        den = (ori_x_sin ** 2 + ori_x_cos ** 2) ** 0.5
        sin_x = ori_x_sin / den
        cos_x = ori_x_cos / den

        ori_f = (cos_x - torch.cos(ori_y)) ** 2 + \
            (sin_x - torch.sin(ori_y)) ** 2
        flags = torch.abs(ori_f) < 1
        ori_loss = (flags == 1).float() * 0.5 * (ori_f ** 2) + \
            (flags == 0).float() * (torch.abs(ori_f) - 0.5)
        if self.reduction == 'mean':
            ori_loss = torch.mean(ori_loss * b_y)
        else:  # self.reduction == 'sum':
            ori_loss = torch.sum(ori_loss * b_y)
            ori_loss /= b_y.size(0)
        return ori_loss

    def getOri(self, ori_x):
        return torch.atan2(ori_x[:, 0], ori_x[:, 1]).unsqueeze(1)

    def getOriXNnm(self):
        # orix 需要的channel数量
        return 2


class CCELoss(torch.nn.Module):
    def __init__(self, b_lambda=1.7, reduction='mean', boundary_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 1.1]):
        super().__init__()
        self.b_lambda = b_lambda
        self.reduction = reduction
        self.boundary_weights = boundary_weights
        # self.ignore_index = 255  # for padded image nyudmt

    def forward(self, b_x, b_y):

        # valid_mask = (b_y != 255)

        b_blance_pos = float(torch.sum(b_y == 1)) / float(torch.numel(b_y))
        b_blance_neg = float(torch.sum(b_y == 0)) / float(torch.numel(b_y))
        b_blance = [b_blance_pos, b_blance_neg]

        b_blance[0] *= 1.1
        weights = torch.zeros(b_y.size(), dtype=b_y.dtype, device=b_y.device)
        weights += (b_y == 1).float() * b_blance[1]
        weights += (b_y == 0).float() * (b_blance[0]) * self.b_lambda


        if isinstance(b_x, list):
            assert len(b_x) == len(self.boundary_weights)

            edge_loss = self.boundary_weights[0] * torch.nn.BCEWithLogitsLoss(weights, reduction=self.reduction)(b_x[0], b_y)
            if self.reduction == 'sum':
                edge_loss /= b_y.size(0)

            for j in range(1, len(self.boundary_weights)):
                c_loss = self.boundary_weights[j] * torch.nn.BCEWithLogitsLoss(weights, reduction=self.reduction)(b_x[j], b_y)
                if self.reduction == 'sum':
                    c_loss /= b_y.size(0)
                
                edge_loss += c_loss
                
        else:

            edge_loss = torch.nn.BCEWithLogitsLoss(weights, reduction=self.reduction)(b_x, b_y)
            if self.reduction == 'sum':
                edge_loss /= b_y.size(0)
        
        return edge_loss
        

class Mask_CCELoss(torch.nn.Module):
    def __init__(self, b_lambda=1.7, reduction='mean', boundary_weights=None):
        super().__init__()
        self.b_lambda = b_lambda
        self.reduction = reduction
        self.boundary_weights = boundary_weights

    def forward(self, b_x, b_y):
        # Create valid mask (ignore pixels with label 255)
        b_y = b_y.squeeze(1)
        valid_mask = (b_y != 255)
        b_y = b_y.float()

        total_valid = valid_mask.sum().item()
        if total_valid == 0:
            # No valid pixels, return zero loss
            return torch.tensor(0.0, device=b_y.device, requires_grad=True)

        # Compute class balance on valid pixels only
        pos_mask = (b_y == 1) & valid_mask
        neg_mask = (b_y == 0) & valid_mask

        pos_ratio = pos_mask.sum().item() / total_valid
        neg_ratio = neg_mask.sum().item() / total_valid

        pos_weight = neg_ratio * 1.1
        neg_weight = pos_ratio * self.b_lambda

        # Create weight map
        weights = torch.zeros_like(b_y)
        weights[pos_mask] = pos_weight
        weights[neg_mask] = neg_weight
        weights = weights * valid_mask  # ignore 255 pixels

        # Loss function with masking
        def masked_bce_loss(pred):
            pred = pred.squeeze(1) if pred.dim() == 4 and pred.size(1) == 1 else pred
            loss = F.binary_cross_entropy_with_logits(pred, b_y, weight=weights, reduction='none')
            loss = loss * valid_mask
            if self.reduction == 'mean':
                return loss.sum() / total_valid
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss

        # Handle multi-scale or single prediction
        if isinstance(b_x, list):
            assert len(b_x) == len(self.boundary_weights)
            total_loss = 0.0
            for j in range(len(b_x)):
                loss = masked_bce_loss(b_x[j]) * self.boundary_weights[j]
                total_loss += loss
        else:
            total_loss = masked_bce_loss(b_x)

        return total_loss




class CCELossMask(torch.nn.Module):
    def __init__(self, b_lambda=1.0, reduction='mean'):
        super().__init__()
        self.b_lambda = b_lambda
        self.reduction = reduction

    def forward(self, b_x, b_y, b_blance, mask):
        b_x = b_x[:, 0]
        b_blance[0] *= 1.1
        weights = torch.zeros(b_y.size(), dtype=b_y.dtype, device=b_y.device)
        weights += (b_y == 1).float() * b_blance[1]
        weights += (b_y == 0).float() * (b_blance[0]) * self.b_lambda
        edge_loss = torch.nn.BCEWithLogitsLoss(weights, reduction=self.reduction)(b_x*mask, b_y*mask)
        # print(b_x.shape , b_y.shape, mask.shape)
        if self.reduction == 'sum':
            edge_loss /= b_y.size(0)
        return edge_loss


class AttentionLoss(nn.Module):
    """
    binary attention loss introduced in DOOBNet https://arxiv.org/pdf/1806.03772.pdf
    extension of focal loss by adding modulating param beta
    """

    def __init__(self, gamma_beta=(0.5, 4), alpha=None, size_average=True, avg_method='batch'):
        """
        :param gamma_beta:
        :param alpha: None or a float
        :param size_average:
        """
        super(AttentionLoss, self).__init__()
        self.gamma = gamma_beta[0]
        self.beta  = gamma_beta[1]
        self.alpha = alpha
        self.size_average = size_average
        self.avg_method   = avg_method
        self.eps = 1e-8

    def forward(self, net_out, target):
        """
        :param net_out:# net_out: (N, 1, H, W) ; activation passed by sigmoid [0~1]
        :param target: (N, 1, H, W)
        :return:
        """
        N, C, H, W = target.shape
        assert net_out.size(1) == 1

        # create mask to identify pixels at boundary
        edge     = (target == 1).float()
        non_edge = (target != 1).float()

        if self.alpha is None:
            alpha = non_edge.sum() / (non_edge.sum() + edge.sum())
        else:
            alpha = self.alpha

        # net_out = torch.sigmoid(net_out)
        net_out = torch.clamp(net_out, self.eps, 1-self.eps)  # according to caffe code

        # in loss_bkp.py
        # loss = -alpha * target * 4 ** ((1.0 - net_out) ** 0.5) * torch.log(net_out + 1e-8) - \
        #        (1.0 - alpha) * (1.0 - target) * 4 ** (net_out ** 0.5) * torch.log(1.0 - net_out + 1e-8)
        # loss_al = torch.sum(loss)

        # scale_edge    = alpha * torch.pow(self.beta, torch.pow((1 - net_out), self.gamma))
        scale_edge = alpha * torch.pow(self.beta, torch.pow((1 - net_out), self.gamma))
        scale_nonedge = (1 - alpha) * torch.pow(self.beta, torch.pow(net_out, self.gamma))

        log_p = net_out.log()
        # log_p = (net_out + self.eps).log()
        log_m_p = (1 - net_out).log()
        # log_m_p = (1 - net_out + self.eps).log()

        loss = - edge * scale_edge * log_p - non_edge * scale_nonedge * log_m_p
        # print(loss)
        # print(loss.sum())
        
        # loss = torch.clamp(loss, self._eps, 1-self._eps)  # according to caffe code
        loss_nan_map = torch.isnan(loss)
        loss[loss_nan_map] = 0.0

        if self.size_average:
            if self.avg_method == 'batch':
                loss = loss.view(N, -1).sum(-1)
            return loss.mean()
        else:
            return loss.sum()  # too big loss may result in too big grad


class FocalLoss(nn.Module):
    """
    focal loss for classification task
    derived from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """
    def __init__(self, gamma=0., alpha='None', size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # default:2 ; for RetinaNet: [0.5, 5]
        self.alpha = alpha  # hyper-param(list) or inverse class frequency
        if isinstance(alpha, (float, int)):  # binary case
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, net_out, target):
        cls_ind = torch.arange(0, net_out.size(1), dtype=torch.long).tolist()
        if net_out.dim() > 2:
            net_out = net_out.view(net_out.size(0), net_out.size(1), -1)  # N,C,H,W => N,C,H*W
            net_out = net_out.transpose(1, 2)  # N,C,H*W => N,H*W,C
            net_out = net_out.contiguous().view(-1, net_out.size(2))  # N,H*W,C => N*H*W,C
        # target = target.view(-1, 1)  # N,H,W => N*H*W,1

        log_pt = F.log_softmax(net_out, dim=1)  # log(softmax(net_out))
        log_pt = log_pt.gather(1, target)  # N*H*W,C => N*H*W,1 gather along axis 1
        log_pt = log_pt.view(-1)  # N*H*W,
        pt = log_pt.data.exp()  # softmax(net_out)

        if self.alpha == 'None':
            # use mini-batch inverse class frequency as alpha
            cls_num = torch.tensor([target.cpu().eq(cls_idx).sum() for cls_idx in cls_ind])
            # alpha = target.cpu().numel() / cls_num.float()  # C,
            alpha = 1 + torch.log(target.cpu().numel() / cls_num.float())  # C,
        else:
            alpha = self.alpha
        if alpha.type() != net_out.data.type():
            alpha = alpha.type_as(net_out.data)

        at = alpha.gather(0, target.data.view(-1))  # N*H*W,
        log_pt = log_pt * at  # alpha * log(softmax(net_out))
        loss = -1 * ((1 - pt) ** self.gamma) * log_pt  # N*H*W,
        if self.size_average:
            return loss.mean()  # average over each loss elem
        else:
            return loss.sum()


class FocalLossV2(nn.Module):
    """
    focal loss for classification task
    derived from https://github.com/CoinCheung/pytorch-loss/blob/master/focal_loss.py
    """

    def __init__(self,
                 alpha=None,
                 gamma=2,
                 reduction='sum',
                 backward_all_edges=False):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-8
        # self.crit = nn.BCEWithLogitsLoss(reduction='none')
        self.backward_all_edges = backward_all_edges

    def forward(self, logits, label):


        # create mask to identify pixels at boundary
        edge     = (label == 1).float()
        non_edge = (label != 1).float()

        if self.alpha is None:
            alpha = non_edge.sum() / (non_edge.sum() + edge.sum())
        else:
            alpha = self.alpha
        
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1-self.eps)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0, F.softplus(logits, -1, 50), logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, -logits + F.softplus(logits, -1, 50), -F.softplus(logits, 1, 50))
        loss = label * alpha * log_probs + (1. - label) * (1. - alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class OcclousionLoss(torch.nn.Module):

    def __init__(self,
                 boundary_weights=[0.5, 0.5, 0.5, 0.5, 1.1, 1.2],
                 boundary_lambda=1.0,
                 orientation_weight=1.1):
        super().__init__()

        self.boundary_weights = boundary_weights
        self.orientation_weight = orientation_weight

        self.b_loss = CCELoss(b_lambda=boundary_lambda, reduction='mean')
        self.o_loss = OORLoss(reduction='mean')

    def forward(self, boundary_x_list, orientation_x, labels):
      
        boundary_y, orientation_y = labels[:, 0], labels[:, 1]
        # print(labels.shape)
        # print(boundary_y.shape)
        # print(boundary_x_list[0].shape)
        # print(len(boundary_x_list), len(self.boundary_weights))
        b_blance_pos = float(torch.sum(boundary_y == 1)) / float(torch.numel(boundary_y))
        b_blance_neg = float(torch.sum(boundary_y == 0)) / float(torch.numel(boundary_y))
        b_blance = [b_blance_pos, b_blance_neg]

        assert len(boundary_x_list) == len(self.boundary_weights)

        boundary_losses = [b_w * self.b_loss(b_x, boundary_y, b_blance)
                           for b_x, b_w in zip(boundary_x_list, self.boundary_weights)]
        # orientation_loss = self.orientation_weight * self.o_loss(orientation_x, orientation_y, boundary_y)

        return boundary_losses, boundary_losses[-1] # orientation_loss
        
        
class OcclousionLossMask(torch.nn.Module):

    def __init__(self,
                 boundary_weights=[0.5, 0.5, 0.5, 0.5, 1.1, 1.2],
                 boundary_lambda=1.0,
                 orientation_weight=1.1,
                 mask_loss_weight=0.2):
        super().__init__()

        self.boundary_weights = boundary_weights
        self.orientation_weight = orientation_weight
        self.mask_loss_weights = [item*mask_loss_weight for item in boundary_weights]

        self.b_loss = CCELoss(b_lambda=boundary_lambda, reduction='mean')
        self.b_loss_mask = CCELossMask(b_lambda=boundary_lambda, reduction='mean')
        self.o_loss = OORLoss(reduction='mean')

    def forward(self, boundary_x_list, orientation_x, labels, mask):
      
        boundary_y, orientation_y = labels[:, 0], labels[:, 1]
        # print(labels.shape)
        # print(boundary_y.shape)
        # print(boundary_x_list[0].shape)
        # print(len(boundary_x_list), len(self.boundary_weights))
        b_blance_pos = float(torch.sum(boundary_y == 1)) / float(torch.numel(boundary_y))
        b_blance_neg = float(torch.sum(boundary_y == 0)) / float(torch.numel(boundary_y))
        b_blance = [b_blance_pos, b_blance_neg]

        assert len(boundary_x_list) == len(self.boundary_weights)

        boundary_losses = [b_w * self.b_loss(b_x, boundary_y, b_blance)
                           for b_x, b_w in zip(boundary_x_list, self.boundary_weights)]
        boundary_losses_mask = [b_w * self.b_loss_mask(b_x, boundary_y, b_blance, mask)
                           for b_x, b_w in zip(boundary_x_list, self.mask_loss_weights)]

        # orientation_loss = self.orientation_weight * self.o_loss(orientation_x, orientation_y, boundary_y)

        return boundary_losses+boundary_losses_mask, boundary_losses[-1] # orientation_loss
        
        
        
class OcclousionFLLoss(torch.nn.Module):

    def __init__(self,
                 boundary_weights=[0.5, 0.5, 0.5, 0.5, 1.1, 1.2],
                 boundary_lambda=1.0,
                 orientation_weight=0.4):
        super().__init__()

        self.boundary_weights = boundary_weights
        self.orientation_weight = orientation_weight

        self.b_loss = CCELoss(b_lambda=boundary_lambda, reduction='mean')
        self.o_loss = FocalLossV2()

    def forward(self, boundary_x_list, orientation_x, labels):
      
        boundary_y, orientation_y = labels[:, 0], labels[:, 1]

        b_blance_pos = float(torch.sum(boundary_y == 1)) / float(torch.numel(boundary_y))
        b_blance_neg = float(torch.sum(boundary_y == 0)) / float(torch.numel(boundary_y))
        b_blance = [b_blance_pos, b_blance_neg]

        assert len(boundary_x_list) == len(self.boundary_weights)

        boundary_losses = [b_w * self.b_loss(b_x, boundary_y, b_blance)
                           for b_x, b_w in zip(boundary_x_list, self.boundary_weights)]
        orientation_loss = self.orientation_weight * self.o_loss(orientation_x, boundary_y)

        return boundary_losses, orientation_loss  
        
        
class FLLoss(torch.nn.Module):

    def __init__(self,
                 boundary_weights=[0.5, 0.5, 0.5, 0.5, 1.1, 1.2],
                 boundary_lambda=1.0,
                 orientation_weight=0.4):
        super().__init__()

        self.boundary_weights = boundary_weights
        self.orientation_weight = orientation_weight

        self.b_loss = FocalLossV2()
        self.o_loss = FocalLossV2()

    def forward(self, boundary_x_list, orientation_x, labels):
      
        boundary_y, orientation_y = labels[:, 0], labels[:, 1]

        assert len(boundary_x_list) == len(self.boundary_weights)

        boundary_losses = [b_w * self.b_loss(b_x, boundary_y)
                           for b_x, b_w in zip(boundary_x_list, self.boundary_weights)]
        orientation_loss = self.orientation_weight * self.o_loss(orientation_x, boundary_y)

        return boundary_losses, orientation_loss                 


# from mtorl.models.losses.isloss import NormalizedFocalLossSigmoid as NFL

# class NormalizedFocalLossSigmoid(nn.Module):
#     def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
#                  from_sigmoid=False, detach_delimeter=True,
#                  batch_axis=0, weight=None, size_average=True,
#                  ignore_label=-1):
#         super(NormalizedFocalLossSigmoid, self).__init__()
#         self._axis = axis
#         self._alpha = alpha
#         self._gamma = gamma
#         self._ignore_label = ignore_label
#         self._weight = weight if weight is not None else 1.0
#         self._batch_axis = batch_axis

#         self._from_logits = from_sigmoid
#         self._eps = eps
#         self._size_average = size_average
#         self._detach_delimeter = detach_delimeter
#         self._max_mult = max_mult
#         self._k_sum = 0
#         self._m_max = 0

#     def forward(self, pred, label):
#         one_hot = label > 0.5
#         sample_weight = label != self._ignore_label

#         if not self._from_logits:
#             pred = torch.sigmoid(pred)
            
#         # pred = torch.clamp(pred, self._eps, 1-self._eps)  # according to caffe code
#         alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
#         pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))
#         # print(pt)
#         beta = (1 - pt) ** self._gamma

#         sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
#         beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
#         mult = sw_sum / (beta_sum + self._eps)
#         if self._detach_delimeter:
#             mult = mult.detach()
#         beta = beta * mult
#         if self._max_mult > 0:
#             beta = torch.clamp_max(beta, self._max_mult)

#         with torch.no_grad():
#             ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
#             sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
#             if np.any(ignore_area == 0):
#                 self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

#                 beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
#                 beta_pmax = beta_pmax.mean().item()
#                 self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

#         loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
#         loss = self._weight * (loss * sample_weight)
        
#         # loss = torch.clamp(loss, self._eps, 1-self._eps)  # according to caffe code
#         loss_nan_map = torch.isnan(loss)
#         loss[loss_nan_map] = 0.0

#         if self._size_average:
#             bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
#             loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
            
#         else:
#             loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

#         return loss

#     def log_states(self, sw, name, global_step):
#         sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
#         sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)


# class SigmoidBinaryCrossEntropyLoss(nn.Module):
#     def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
#         super(SigmoidBinaryCrossEntropyLoss, self).__init__()
#         self._from_sigmoid = from_sigmoid
#         self._ignore_label = ignore_label
#         self._weight = weight if weight is not None else 1.0
#         self._batch_axis = batch_axis

#     def forward(self, pred, label):
#         label = label.view(pred.size())
#         sample_weight = label != self._ignore_label
#         label = torch.where(sample_weight, label, torch.zeros_like(label))

#         if not self._from_sigmoid:
#             loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
#         else:
#             eps = 1e-12
#             loss = -(torch.log(pred + eps) * label
#                      + torch.log(1. - pred + eps) * (1. - label))

#         loss = self._weight * (loss * sample_weight)
#         # loss = torch.clamp(loss, self._eps, 1-self._eps)  # according to caffe code
#         # loss_nan_map = torch.isnan(loss)
#         # loss[loss_nan_map] = 0.0
#         return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


# class OcclousionLoss(torch.nn.Module):

#     def __init__(self,
#                  boundary_weights=[0.5, 0.5, 0.5, 0.5, 1.1, 1.2],
#                  boundary_lambda=1.0,
#                  orientation_weight=1.1):
#         super().__init__()

#         self.boundary_weights = boundary_weights
#         self.orientation_weight = orientation_weight

#         self.b_loss = NFL(alpha=0.5, gamma=2) 
#         self.o_loss = OORLoss(reduction='mean')

#     def forward(self, boundary_x_list, orientation_x, labels):
      
#         boundary_y, orientation_y = labels[:, 0], labels[:, 1]
#         b_blance_pos = float(torch.sum(boundary_y == 1)) / float(torch.numel(boundary_y))
#         b_blance_neg = float(torch.sum(boundary_y == 0)) / float(torch.numel(boundary_y))
#         b_blance = [b_blance_pos, b_blance_neg]

#         assert len(boundary_x_list) == len(self.boundary_weights)

#         boundary_losses = [b_w * self.b_loss(b_x, boundary_y)
#                            for b_x, b_w in zip(boundary_x_list, self.boundary_weights)]

#         return boundary_losses, boundary_losses[-1] 
