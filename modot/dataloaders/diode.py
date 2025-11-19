import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import cv2
import numpy as np
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

            
class DIODE(Dataset):

    def __init__(self,
                 root=None,
                 split_file='val',
                 transform=None,
                 retname=True,
                 dataset="diode",
                 model_name="other",
                 **kwargs):


        self.root = root # + "/NYUDv2/"
        self.transform = transform
        self.retname = retname
        self.model_name = model_name

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

                if dataset == "diode":
                    self.im_ids.append(item.split("/")[-1].replace(".png", ""))
                    _image = self.root + item
                    _edge = self.root + item.replace("/imgs/","/gt/")
                elif dataset == "entityseg":
                    self.im_ids.append(item.split("/")[-1]) 
                    _image = self.root + item + ".jpg"
                    _edge = self.root + item.replace("imgs/","/gt/") + ".png" 

                assert os.path.isfile(_image)
                self.images.append(_image)



                assert os.path.isfile(_edge)
                self.edges.append(_edge) 
              
                _depth = self.root +  f'depth/{item}.npy'
                self.depths.append(_depth)


        # Display stats
        # print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        _edge =  transfer_label(self.edges[index]) # Image.open(self.edges[index]).convert('L')
        _edge = np.expand_dims(np.array(_edge, dtype=np.float32, copy=False), axis=2)  # / 255.

        sample['edge'] = _edge
        
        if self.retname:
            sample['meta'] = {'name': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}


        if self.transform is not None:
            sample = self.transform(sample)
        
        # # print(torch.max(sample['depth']), torch.min(sample['depth']))


        return sample


    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        if self.model_name == "invpt":
            _img = _img.resize((1024,768))
        _img = np.array(_img, dtype=np.float32, copy=False)
        return _img

    def _load_edge(self, index):
        _edge = Image.open(self.edges[index])
        print(np.array(_edge).shape)
        _edge = np.expand_dims(np.array(_edge, dtype=np.float32, copy=False), axis=2)  # / 255.
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
        return 'Depth-OB (split=' + str(self.split) + ')'




def transfer_label(img_paths, is_ob3=False, left_rule=True, keep_n180=True, angle_degree=False):

    # counter_list = list()
    # BGR is ok
    img_ob_path, img_oo_path = img_paths,img_paths
    img_oo = cv2.imread(img_oo_path) # , cv2.IMREAD_UNCHANGED)
    if is_ob3:
        img_ob_path = img_ob_path.replace("OB", "OB3")
        img_ob = cv2.imread(img_ob_path)
    else:
        img_ob = cv2.imread(img_ob_path)

    img_label = np.zeros((img_ob.shape[0], img_ob.shape[1], 2))

    for i in range(img_ob.shape[0]):
        for j in range(img_ob.shape[1]):
        
            # ob, ob3 to be finished
            if img_ob[i, j, :].tolist() != [0, 0, 0]: # == [255, 255, 255]:
                img_label[i, j, 0] = 1

            # oo
            # if img_oo[i, j, :].tolist() != [112, 112, 112]:
                angle = img_oo[i, j, 0]
                # image/screen coordinates angle
                angle = int(angle * 360 / 255)
                # one easy way is not using opencv img store oo info, use numpy npy.
                # angle_lists = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
                if angle == 269 or angle == 224 or angle == 314:
                    angle += 1
                # if we subtract 90 degree, it follows the left rule
                if left_rule:
                    angle = angle - 90
                if not left_rule and keep_n180 and angle == 360:
                    # special angle, cf. code geocc
                    angle = -180
                if angle > 180:
                    # [0, 360] -> [-180, 180]
                    angle = angle - 360
                # counter_list.append(angle)
                if angle_degree:
                    img_label[i, j, 1] = angle / 180
                else:
                    img_label[i, j, 1] = angle * PI / 180                


    return np.array(img_label[:,:, 0])




def get_gt_image(img_paths, is_ob3=False):
    """
    BGR is ok
    """
    img_ob_path, img_oo_path = img_paths
    img_oo = cv2.imread(img_oo_path)
    if is_ob3:
        img_ob_path = img_ob_path.replace("OB", "OB3")
        img_ob = cv2.imread(img_ob_path)
    else:
        img_ob = cv2.imread(img_ob_path)
    return img_ob, img_oo

