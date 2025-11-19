import cv2
import time 
import os
import numpy as np
from collections import Counter
import pdb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# from sklearn.metrics import precision_recall_curve
# from edgeEvalPy.nms_process import nms_process_one_image

from utils import compute_errors
from skimage import feature
from scipy import ndimage


def compute_edge_metrics(pred_prob, gt_edge, threshold=0.7, edge_nms=False):
    """
    Compute evaluation metrics for edge prediction.

    Parameters:
    - pred_prob (np.ndarray): Predicted edge probability map (values between 0 and 1).
    - gt_edge (np.ndarray): Ground truth edge map (binary: 0 or 1).
    - threshold (float): Threshold to binarize the predicted probabilities.

    Returns:
    - metrics (dict): Dictionary containing TP, FP, TN, FN, Precision, Recall, Accuracy, F1-Score.
    """

    if edge_nms:
        pred_edge = nms_process_one_image(pred_prob, None, False) / 255


    # Binarize the predicted edge map based on the threshold
    pred_edge = (pred_prob >= threshold).astype(int)
      
    # Flatten the arrays to 1D for metric calculations
    pred_flat = pred_edge.flatten()
    gt_flat = gt_edge.flatten()
    
    # print(Counter(pred_flat))
    # print(Counter(gt_flat))
    # Calculate True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum((pred_flat == 1) & (gt_flat == 1))
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))
    TN = np.sum((pred_flat == 0) & (gt_flat == 0))
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))
   
    # Calculate Precision, Recall, Accuracy, F1-Score
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    accuracy = accuracy_score(gt_flat, pred_flat)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)
   
    metrics = {
        'Threshold': threshold,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'F1-Score': f1
    }
   
    return metrics

def compute_edge_metrics_multiple_thresholds(pred_prob, gt_edge, thresholds=None):
    """
    Compute evaluation metrics across multiple thresholds.

    Parameters:
    - pred_prob (np.ndarray): Predicted edge probability map (values between 0 and 1).
    - gt_edge (np.ndarray): Ground truth edge map (binary: 0 or 1).
    - thresholds (list or np.ndarray): List of threshold values to evaluate.

    Returns:
    - metrics_list (list): List of dictionaries containing metrics for each threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, num=21)  # Default: thresholds from 0 to 1 in steps of 0.05
   
    metrics_list = []
    for thresh in thresholds:
        metrics = compute_metrics(pred_prob, gt_edge, threshold=thresh)
        metrics_list.append(metrics)
   
    return metrics_list


def compute_depth_metrics(input, target, mask=None):
    if mask is None:
        rmse = np.sqrt(np.mean((input - target) ** 2))
        rmse_log = np.sqrt(np.mean(np.log10(np.clip(input, a_min=1e-12, a_max=1e12)) - np.log10(
            np.clip(target, a_min=1e-12, a_max=1e12))) ** 2)

        avg_log10 = np.mean(
            np.abs(
                np.log10(np.clip(input, a_min=1e-12, a_max=1e12)) - np.log10(np.clip(target, a_min=1e-12, a_max=1e12))))

        rel = np.mean(np.abs(input - target) / target)
    else:
        N = np.sum(mask)

        diff = mask * (input - target)
        diff = diff ** 2
        diff_log = mask * (np.log(np.clip(input, a_min=1e-12, a_max=1e12)) - np.log(
            np.clip(target, a_min=1e-12, a_max=1e12))) ** 2
        mse = np.sum(diff)
        mse_log = np.sum(diff_log)
        rmse = np.sqrt(float(mse) / N)
        rmse_log = np.sqrt(float(mse_log) / N)

        avg_log10 = np.sum(
            mask * np.abs(np.log10(np.clip(input, a_min=1e-12, a_max=1e8))
                          - np.log10(np.clip(target, a_min=1e-8, a_max=1e8))))
        avg_log10 = float(avg_log10) / N

        rel = float(np.sum(np.abs(input - target) / target)) / N

    acc_map = np.max((target / (input + 1e-8), input / (target + 1e-8)), axis=0)
    acc_1_map = acc_map < 1.25
    acc_2_map = acc_map < 1.25 ** 2
    acc_3_map = acc_map < 1.25 ** 3
    if mask is not None:
        acc_1_map[mask == 0] = False
        acc_2_map[mask == 0] = False
        acc_3_map[mask == 0] = False

        N = np.sum(mask)
    else:
        N = np.prod(input.shape)

    acc_1 = len(acc_1_map[acc_1_map == True]) / N
    acc_2 = len(acc_2_map[acc_2_map == True]) / N
    acc_3 = len(acc_2_map[acc_3_map == True]) / N

    return acc_1, acc_2, acc_3, rel, avg_log10, rmse, rmse_log




def compute_depth_boundary_error(edges_gt, pred, mask=None, low_thresh=0.15, high_thresh=0.3):
    # skip dbe if there is no ground truth distinct edge
    if np.sum(edges_gt) == 0:
        dbe_acc = np.nan
        dbe_com = np.nan
        edges_est = np.empty(pred.shape).astype(int)
    else:

        # normalize est depth map from 0 to 1
        pred_normalized = pred.copy().astype('f')
        pred_normalized[pred_normalized == 0] = np.nan
        pred_normalized = pred_normalized - np.nanmin(pred_normalized)
        pred_normalized = pred_normalized / np.nanmax(pred_normalized)

        # apply canny filter
        edges_est = feature.canny(pred_normalized, sigma=np.sqrt(2), low_threshold=low_thresh,
                                  high_threshold=high_thresh)

        # compute distance transform for chamfer metric
        D_gt = ndimage.distance_transform_edt(1 - edges_gt)
        D_est = ndimage.distance_transform_edt(1 - edges_est)

        max_dist_thr = 10.  # Threshold for local neighborhood

        mask_D_gt = D_gt < max_dist_thr  # truncate distance transform map

        E_fin_est_filt = edges_est * mask_D_gt  # compute shortest distance for all predicted edges
        if mask is None:
            mask = np.ones(shape=E_fin_est_filt.shape)
        E_fin_est_filt = E_fin_est_filt * mask
        D_gt = D_gt * mask

        if np.sum(E_fin_est_filt) == 0:  # assign MAX value if no edges could be detected in prediction
            dbe_acc = max_dist_thr
            dbe_com = max_dist_thr
        else:
            # accuracy: directed chamfer distance of predicted edges towards gt edges
            dbe_acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(E_fin_est_filt)

            # completeness: sum of undirected chamfer distances of predicted and gt edges
            ch1 = D_gt * edges_est  # dist(predicted,gt)
            ch1[ch1 > max_dist_thr] = max_dist_thr  # truncate distances
            ch2 = D_est * edges_gt  # dist(gt, predicted)
            ch2[ch2 > max_dist_thr] = max_dist_thr  # truncate distances
            res = ch1 + ch2  # summed distances
            dbe_com = np.nansum(res) / (np.nansum(edges_est) + np.nansum(edges_gt))  # normalized

    return dbe_acc, dbe_com

   


if __name__ == '__main__':
    pass
