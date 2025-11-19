import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import cv2
import numpy as np
from scipy import io
from PIL import Image, ImageOps
import os
import random
from collections import Counter
PI = np.pi
from utils import DistributedSamplerNoEvenlyDivisible


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def n_preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

            
class IBIMS(Dataset):

    def __init__(self,
                 root=None,
                 split_file='val',
                 transform=None,
                 retname=True,
                 **kwargs):


        self.root = root # + "/NYUDv2/"
        self.transform = transform
        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
    
        # OB
        self.edges = []

        # Depth
        self.depths = []
        self.dvd_value = 1
        print()
        print("All depth values divides ", self.dvd_value)
        print()


        with open(split_file, 'r') as f:
            lines = f.read().splitlines()

            for item in lines:

                self.im_ids.append(item)

                # Images
                _image = self.root + f"/imgs/{item}.png"
                # print(_image)
                assert os.path.isfile(_image)
                self.images.append(_image)


                _edge = self.root + f"/ob/{item}.png"
                # print(_edge)
                assert os.path.isfile(_edge)
                self.edges.append(_edge) 
              
                _depth = self.root +  f'/mats/{item}.mat'
                # assert os.path.isfile(_depth)
                self.depths.append(_depth)


        # Display stats
        # print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        _edge = self._load_edge(index)
        sample['edge'] = _edge

        # copy from P2ORM, differ depth anything
        _depth, dmask_valid, d_edge = self._load_depths_from_mat(self.depths[index])
        _depth = _depth / self.dvd_value
        sample['depth'] =  _depth
        
        if self.retname:
            sample['meta'] = {'name': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1]),
                              "d_valid_mask": dmask_valid }

        if self.transform is not None:
            sample = self.transform(sample)
    

        return sample


    def _load_depths_from_mat(self, gt_mat):

        # load ground truth depth
        # image_data = io.loadmat(gt_mat)
        # print(image_data.keys())
        # gt = image_data['gtStruct'][0]
        # print(gt.shape)

        image_data = io.loadmat(gt_mat.replace("mats", "gt_depth"))
        # print(image_data.keys())
        data = image_data['data']

        # extract neccessary data
        depth = data['depth'][0][0]  # Raw depth map
        mask_invalid = data['mask_invalid'][0][0]  # Mask for invalid pixels
        mask_transp = data['mask_transp'][0][0]  # Mask for transparent pixels
        edge = data['edges'][0][0]

        mask_missing = depth.copy()  # Mask for further missing depth values in depth map
        mask_missing[mask_missing != 0] = 1

        mask_valid = mask_transp * mask_invalid * mask_missing  # Combine masks

        depth_valid = depth * mask_valid

        depth_valid = np.expand_dims(depth_valid.astype(np.float32), axis=2)

        return depth_valid, mask_valid, edge


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
        return 'Multitask (split=' + str(self.split) + ')'




