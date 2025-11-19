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
import torch

from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from six.moves import urllib

from collections import Counter


def merge_two_gt(ob, edge, save_name=None):
    assert ob.shape == edge.shape
    for i in range(ob.shape[0]):
        for j in range(ob.shape[1]):
            if ob[i,j] != 0:
                edge[i,j] = 1
    return edge
    # cv2.imwrite(save_name, edge)


class NYUDMT(data.Dataset):

    def __init__(self,
                 root=None,
                 split_file='val',
                 transform=None,
                 retname=True,
                 **kwargs):


        self.root = root + "/NYUDv2/"
        self.transform = transform
        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
    
        # OB
        self.edges = []

        # Depth
        self.depths = []
        self.dvd_value = 1000
        print()
        print("All depth values divides ", self.dvd_value)
        print()



        with open(split_file, 'r') as f:
            lines = f.read().splitlines()

            for item in lines:

                name = int(item) + 1
                item = str(name).zfill(4)

                self.im_ids.append(item)

                # Images
                _image = self.root + f'images/{item}.png'
                # print(_image)
                assert os.path.isfile(_image)
                self.images.append(_image)

                # OBs
                _edge = self.root +  f'edge/{item}.png'
                assert os.path.isfile(_edge)
                self.edges.append(_edge) 
              
                _depth = self.root +  f'depth/{item}.npy'
                assert os.path.isfile(_depth)
                self.depths.append(_depth)


        # Display stats
        # print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        _edge = self._load_edge(index)
        sample['edge'] = _edge

        _depth = self._load_depth(index)
        # _depth = _depth / self.dvd_value
        sample['depth'] = _depth

        
        if self.retname:
            sample['meta'] = {'name': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        # print(np.max(sample['depth']), np.min(sample['depth']))

        if self.transform is not None:
            sample = self.transform(sample)
        
        # print(torch.max(sample['depth']), torch.min(sample['depth']))
        # print(sample['depth'].shape)

        # pdb.set_trace()

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

    def _load_semseg(self, index):
        # Note: We ignore the background class (40-way classification), as in related work:
        _semseg = Image.open(self.semsegs[index])
        _semseg = np.expand_dims(np.array(_semseg, dtype=np.float32, copy=False), axis=2) - 1
        _semseg[_semseg == -1] = 255
        return _semseg

    def _load_depth(self, index):
        _depth = np.load(self.depths[index])
        _depth = np.expand_dims(_depth.astype(np.float32), axis=2)
        return _depth

    def _load_normals(self, index):
        _normals = Image.open(self.normals[index])
        _normals = 2 * np.array(_normals, dtype=np.float32, copy=False) / 255. - 1
        return _normals

    def __str__(self):
        return 'NYUD Multitask (split=' + str(self.split) + ')'




