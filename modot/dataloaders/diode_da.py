# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x : x
        self.resize = transforms.Resize(480)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        image = self.resize(image)

        return {'image': image, 'depth': depth, 'dataset': "diode"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
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


class DIODE_DA(Dataset):
    # from depth anything
    def __init__(self,                  
                 root=None,
                 split_file='val',
                 transform=None,
                 retname=True,):


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
        self.depth_masks = []
        # self.dvd_value = 1000
        # print()
        # print("All depth values divides ", self.dvd_value)
        # print()

        with open(split_file, 'r') as f:
            lines = f.read().splitlines()

            for item in lines:
                
                item_split = item.replace(".png", "").split("/")
                ids_name = f"{item_split[0]}_{item_split[1]}_{item_split[2]}_{item_split[3]}"

                self.im_ids.append(ids_name)

                # Images
                _image = self.root + item

                # print(_image)
                assert os.path.isfile(_image)
                self.images.append(_image)

                self.edges.append(0) 
              
                _depth = self.root +  item.replace(".png", "_depth.npy")

                self.depths.append(_depth)

                _depth_mask = self.root +  item.replace(".png", "_depth_mask.npy")

                self.depth_masks.append(_depth_mask)


        # Display stats
        # print('Number of dataset images: {:d}'.format(len(self.images)))



    def __getitem__(self, idx):

        image_path = self.images[idx]
        depth_path = self.depths[idx]
        depth_mask_path = self.depth_masks[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.load(depth_path)  # in meters
        valid = np.load(depth_mask_path)  # binary

        # depth[depth > 8] = -1
        valid = valid[..., None]

        sample = dict(image=image, depth=depth, valid=valid)

        # print(image.shape, depth.shape, valid.shape)
        if self.retname:
                sample['meta'] = {'name': str(self.im_ids[idx]),
                                  'im_size': (image.shape[0], image.shape[1])}

        # print(np.max(sample['depth']), np.min(sample['depth']))

        if self.transform is not None:
            sample = self.transform(sample)
            

        return sample



    def __len__(self):
        return len(self.images)


def get_diode_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = DIODE(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)

# get_diode_loader(data_dir_root="datasets/diode/val/outdoor")
