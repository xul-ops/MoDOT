# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import sys
import tarfile
import cv2
import pdb
import h5py
import random

from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from six.moves import urllib

from collections import Counter


def read_hdf5_depth(filepath):
    with h5py.File(filepath, 'r') as f:
        distance_img_meters = f['dataset'][:]  # .astype(np.float32)    
    # print(type(distance_img_meters))
    # print(distance_img_meters.shape)
    # print(distance_img_meters.max(), distance_img_meters.min())
    # pdb.set_trace()
    depth = distance_img_meters # * 1000
    # print(depth.max(), depth.min())
    # print(Counter(depth.reshape(-1)))
    return depth


def read_hdf5_normal(filepath):
    with h5py.File(filepath, 'r') as f:
        gt = f['dataset'][:].astype(np.float32)    

    # note: every valid normal vector in the dataset is already normalized to
    # have unit length

    # convert back to uint8, clipping is necessary to avoid over/-underflows
    normal = gt # ((gt + 1)*127).clip(0, 255).astype('uint8')

    # format 
    return normal


def read_hdf5_semseg(filepath):
    with h5py.File(filepath, 'r') as f:
        gt = f['dataset'][:].astype(np.int32)  # .astype(np.float32)    
    # print(type(distance_img_meters))
    # print(distance_img_meters.shape)
    # print(distance_img_meters.max(), distance_img_meters.min())
    # pdb.set_trace()
    # gt[gt == -1] = 0
    semseg = gt
    # print(depth.max(), depth.min())
    # print(Counter(depth.reshape(-1)))
    return semseg


def merge_two_gt(ob, edge, save_name=None):
    assert ob.shape == edge.shape
    for i in range(ob.shape[0]):
        for j in range(ob.shape[1]):
            if ob[i,j] != 0:
                edge[i,j] = 1
    return edge
    # cv2.imwrite(save_name, edge)



def distance_to_zdepth(distance, fx=886.81, fy=927.06, cx=512, cy=384, width=1024, height=768):
    """
    Convert Euclidean distance-to-camera predictions into z-buffer depth.

    Args:
        distance: (B, H, W) tensor, predicted Euclidean distance (in meters).
        fx, fy: focal lengths (pixels).
        cx, cy: principal point (pixels).
        width, height: image size.

    Returns:
        zdepth: (B, H, W) tensor, z-buffer depth (in meters).
    """
    device = distance.device
    B, H, W = distance.shape

    # pixel grid
    u = torch.arange(W, device=device).view(1, 1, W).expand(1, H, W)
    v = torch.arange(H, device=device).view(1, H, 1).expand(1, H, W)

    # normalize relative to principal point
    x = (u - cx) / fx   # (1,H,W)
    y = (v - cy) / fy   # (1,H,W)

    # ray length factor
    ray_norm = torch.sqrt(x**2 + y**2 + 1.0)  # (1,H,W)

    # convert distance -> z depth
    zdepth = distance / ray_norm  # (B,H,W)

    return zdepth


class Hypersim(data.Dataset):
    """
    Hypersim subset
    ['ai_012_007', 'cam_01', '0000']  contains all NaN values.

    """

    def __init__(self,
                 root=None,
                 split_file='val',
                 transform=None,
                 retname=True,
                 **kwargs):


        self.root = root
        self.transform = transform
        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
    
        # OB
        self.edges = []
        self.instance_contours = []

        # Depth
        self.depths = []
        self.dvd_value = 3.05
        print()
        print("All depth values divides ", self.dvd_value)
        print()

        with open(split_file, 'r') as f:
            lines = f.read().splitlines()

            for item in lines:
                c_info = item.split("/")

                self.im_ids.append(f"{c_info[1]}_{c_info[2]}_{c_info[-1]}")
                c_root_path = root + f"/{c_info[0]}/{c_info[1]}/{c_info[2]}/"

                # Images
                _image = c_root_path + f'/tonemap/{c_info[-1]}.jpg'
                assert os.path.isfile(_image)
                self.images.append(_image)

                # OBs
                _edge = c_root_path +  f'/ob/{c_info[-1]}.png'
                assert os.path.isfile(_edge)
                self.edges.append(_edge) 

                _contour = c_root_path +  f'/instance_contour/{c_info[-1]}.png'
                assert os.path.isfile(_contour)
                self.instance_contours.append(_contour)

                # Depth
                _depth = c_root_path +  f'/depth_hdf5/{c_info[-1]}.hdf5'
                assert os.path.isfile(_depth)
                self.depths.append(_depth)

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        _edge = self._load_edge(index)
        _contour = self._load_contour(index)
        _edge = merge_two_gt(_edge, _contour)
        sample['edge'] = _edge


        _depth = self._load_depth(index)
        _depth = _depth / self.dvd_value        
        sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'name': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _img = np.array(_img, dtype=np.float32, copy=False)
        return _img

    def _load_edge(self, index):
        _edge = Image.open(self.edges[index])
        _edge = np.expand_dims(np.array(_edge, dtype=np.float32, copy=False), axis=2) / 255.
        return _edge

    def _load_contour(self, index):
        _edge = Image.open(self.instance_contours[index])
        _edge = np.expand_dims(np.array(_edge, dtype=np.float32, copy=False), axis=2) / 255.
        return _edge

    def _load_semseg(self, index):
        # Note: We ignore the background class (40-way classification), as in related work:
        _semseg = read_hdf5_semseg(self.semsegs[index])  # Image.open(self.semsegs[index])
        _semseg = np.expand_dims(np.array(_semseg, dtype=np.float32, copy=False), axis=2) - 1
        _semseg[_semseg == -1] = 255
        return _semseg

    def _load_depth(self, index):
        # this is distance, not z-depth.
        # to transform GT value to z-depth, use the transform function from depth anything (in top of this file)
        # in supp, when compare with DA, the values used for comparing are z-depth
        if self.depths[index].endswith(".hdf5"):
            _depth = read_hdf5_depth(self.depths[index]) #  * 1000  
        else:
            _depth = np.array(Image.open(self.depths[index]).convert("L"))
        _depth = np.expand_dims(_depth.astype(np.float32), axis=2)
        return _depth

    def _load_normals(self, index):
        _normals = read_hdf5_normal(self.normals[index]) # Image.open(self.normals[index])
        _normals = 2 * np.array(_normals, dtype=np.float32, copy=False) / 255. - 1
        return _normals

    def __str__(self):
        return 'Hypersim Multitask (split=' + str(self.split) + ')'



def normalize_depth_image(depth_image, min_value=0, max_value=1):
    # Get the minimum and maximum depth values, ignoring NaNs
    valid_pixels = depth_image[~np.isnan(depth_image)]
    
    if valid_pixels.size > 0:
        min_depth = np.min(valid_pixels)
        max_depth = np.max(valid_pixels)
        
        # Normalize only valid pixels (ignore NaNs)
        normalized_image = (depth_image - min_depth) / (max_depth - min_depth)
        
        # Rescale to the desired range (default is [0, 1])
        normalized_image = normalized_image * (max_value - min_value) + min_value
        
        # Keep NaN values unchanged
        normalized_image[np.isnan(depth_image)] = np.nan
        
        return normalized_image
    else:
        # If the image is all NaN, return the original image
        return depth_image

