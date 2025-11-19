import torch
from torch.utils.data import Dataset
import torch.utils.data.distributed
from torchvision import transforms

import pdb
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import random
from collections import Counter
PI = np.pi


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


            
class OB_FUTURE(Dataset):
    def __init__(self,                  
                 root=None,
                 split_file="",
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

        # Depth
        self.depths = []
        self.dvd_value = 25.5
        print()
        print("All depth values divides ", self.dvd_value)
        print()

        with open(split_file, 'r') as f:
            lines = f.read().splitlines()
            if split_file.endswith("_1p.txt"):
                remove_scenes = ['01219', '05695', '11118', '17595']
                lines.remove(remove_scenes[0])
                lines.remove(remove_scenes[1])
                lines.remove(remove_scenes[2])
                lines.remove(remove_scenes[3])

            for item in lines:
                c_info = item

                self.im_ids.append(c_info)

                c_root_path = root + c_info + "/"

                _image = c_root_path + 'rgb_1.png'
                assert os.path.isfile(_image)
                self.images.append(_image)

                _edge = c_root_path +  'dis_fOB.png'
                assert os.path.isfile(_edge)
                self.edges.append(_edge)

                _depth = c_root_path +  'bdepth.png'
                assert os.path.isfile(_depth)
                self.depths.append(_depth)

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))        



    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index) # Image.open(self.images[index]).convert('RGB')  
        sample['image'] = _img


        _edge = self._load_edge(index)
        sample['edge'] = _edge

        _depth = self._load_depth(index)
        sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'img_name': str(self.im_ids[index]),
                              'img_size': (_img.shape[0], _img.shape[1])}

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
        _edge = np.array(Image.open(self.edges[index]).convert("L")) 
        _edge[_edge != 0] = 1
        _edge = np.expand_dims(np.array(_edge, dtype=np.float32, copy=False), axis=2)                       
        return _edge # .astype(np.float32)

    def _load_contour(self, index):
        _edge = Image.open(self.instance_contours[index])
        _edge = np.array(_edge) / 255. # np.expand_dims(np.array(_edge, dtype=np.float32, copy=False), axis=2) / 255.
        _edge = np.expand_dims(np.array(_edge, dtype=np.float32, copy=False), axis=2) / 255.        
        return _edge # .astype(np.float32)

    def _load_semseg(self, index):
        # Note: We ignore the background class (40-way classification), as in related work:
        _semseg = read_hdf5_semseg(self.semsegs[index])  # Image.open(self.semsegs[index])
        _semseg = np.expand_dims(np.array(_semseg, dtype=np.float32, copy=False), axis=2) - 1
        _semseg[_semseg == -1] = 255
        return _semseg

    def _load_depth(self, index):
        _depth = np.array(Image.open(self.depths[index]).convert("L")) / self.dvd_value
        
        _depth = np.expand_dims(_depth.astype(np.float32), axis=2)
        return _depth # .astype(np.float32)

    def _load_normals(self, index):
        _normals = read_hdf5_normal(self.normals[index]) # Image.open(self.normals[index])
        _normals = 2 * np.array(_normals, dtype=np.float32, copy=False) / 255. - 1
        return _normals

    def __str__(self):
        return 'OB_FUTURE Multitask (split=' + str(self.split_file) + ')'



class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        ob_gt = sample['ob']
        ob_gt = self.to_tensor(ob_gt)

        if self.mode == 'test':
            # ob_gt = torch.from_array(sample['ob'])
            return {'image': image, 'focal': focal, "ob": ob_gt}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            # ob_gt = self.to_tensor(ob_gt)
            return {'image': image, 'depth': depth, 'focal': focal, "ob": ob_gt}
        else:
            # ob_gt = torch.from_array(sample['ob'])
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, 'name': sample['name'], "ob": ob_gt}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img



def transfer_label(img_paths, is_ob3=False, left_rule=True, keep_n180=True, angle_degree=False):

    # counter_list = list()
    # BGR is ok
    img_ob_path, img_oo_path = img_paths
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


    return img_label


def init_gt_files(root_path, img_list_path, dataset_name, train_copy=False):
    """
    this function can be a class staticmethod
    """
    assert dataset_name in ["synocc", "synocc_rgba", "cmu", "piod", "bsds", "nyuocpp", "diode", 'entityseg']
    
    if dataset_name == "synocc":
        with open(img_list_path, 'r') as f:
            names = f.readlines()

        names = [x.replace('\n', '') for x in names]
        # img_list = [os.path.join(root_path, name, 'brgba.png') for name in names]
        img_list = [os.path.join(root_path, name, 'rgb_1.png') for name in names]
        # dis_fOB.png
        gt_list = [[os.path.join(root_path, name, 'dis_fOB.png'),
                    os.path.join(root_path, name, 'bdepth.png')]
                   for name in names]
        img_names = names
        return img_list, gt_list, img_names

    elif dataset_name == "nyuocpp":
        with open(img_list_path, 'r') as f:
            names = f.readlines()
        names = [x.replace('\n', '') for x in names]
        img_list = [os.path.join(root_path, 'imgs/'+name) for name in names]
        gt_list = [[os.path.join(root_path, 'ob/'+name),
                    os.path.join(root_path, 'oo/'+name)]
                   for name in names]
        img_names = [x.replace('.png', '') for x in names]
        if train_copy:
            img_list_1, gt_list_1, img_names_1 = img_list.copy(), gt_list.copy(), img_names.copy()
            copy_time = 16
            for i in range(copy_time):
                img_list.extend(img_list_1)
                gt_list.extend(gt_list_1)
                img_names.extend(img_names_1)        
        return img_list, gt_list, img_names
    
    elif dataset_name == "diode":
        with open(img_list_path, 'r') as f:
            names = f.readlines()
        names = [x.replace('\n', '') for x in names]
        img_names = [x.split('/')[-1] for x in names]
        img_list = [os.path.join(root_path, name) for name in names]
        gt_list = [[os.path.join(root_path, 'gt/'+name),
                    os.path.join(root_path, 'gt/'+name)]
                   for name in img_names]
        img_names_ = [x.replace('.png', '') for x in img_names]
        return img_list, gt_list, img_names_

               
    else:
        print("Add new occ dataset.")
