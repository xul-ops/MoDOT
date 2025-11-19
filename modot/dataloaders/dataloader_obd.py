import torch
from torch.utils.data import Dataset, DataLoader
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
from utils import DistributedSamplerNoEvenlyDivisible

from .synocc import OB_FUTURE
from .hypersim import Hypersim
# from .nyud import NYUD
from .nyudmt import NYUDMT
# from .nyudmtR import NYUDMTR
from .diode import DIODE
from .ibims import IBIMS
# from .openworld import OpenWorld
from .diode_da import DIODE_DA


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})



class NewDataLoader(object):
    def __init__(self, args, mode, preprocessing_transforms=None):
        # print(preprocessing_transforms)

        if mode == 'train':
            if args.dataset == "synocc":
                self.training_samples = OB_FUTURE(root=args.data_path, split_file=args.filenames_file, transform=preprocessing_transforms, retname=True)

            elif args.dataset == "hypersim":
                self.training_samples = Hypersim(root=args.data_path, split_file=args.filenames_file, transform=preprocessing_transforms, retname=True, 
                                                )
            elif args.dataset == "nyudmt":
                self.training_samples = NYUDMT(root=args.data_path, split_file=args.filenames_file, transform=preprocessing_transforms, retname=True,
                                                )            
            # elif args.dataset == "nyudmt_reverse":
            #     self.training_samples = NYUDMTR(root=args.data_path, split_file=args.filenames_file, transform=preprocessing_transforms, retname=True,
            #                                     using_ob=args.hypersim_add_contour)     
            else:
                raise ValueError

            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.train_sampler = None
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':

            if args.dataset_eval == "synocc":
                self.testing_samples = OB_FUTURE(root=args.data_path_eval, split_file=args.filenames_file_eval, transform=preprocessing_transforms, retname=True)

            elif args.dataset_eval == "hypersim":
                self.testing_samples = Hypersim(root=args.data_path_eval, split_file=args.filenames_file_eval, transform=preprocessing_transforms, retname=True, 
                                                    add_contour=args.hypersim_add_contour, use_hdf5=args.hypersim_hdf5, b_type=args.hypersim_b_type)

            elif args.dataset_eval in ["diode", "entityseg"]:
                self.testing_samples = DIODE(root=args.data_path_eval, split_file=args.filenames_file_eval, transform=preprocessing_transforms, retname=True,
                                              dataset=args.dataset_eval, model_name=args.model_name )

            elif args.dataset_eval == "ibims":
                self.testing_samples = IBIMS(root=args.data_path_eval, split_file=args.filenames_file_eval, transform=preprocessing_transforms, retname=True,
                                                )

            elif args.dataset_eval == "nyudmt":
                self.testing_samples = NYUDMT(root=args.data_path_eval, split_file=args.filenames_file_eval, transform=preprocessing_transforms, retname=True,
                                                using_ob=args.hypersim_add_contour)

            # elif args.dataset_eval == "nyud":
            #     self.testing_samples = NYUD(root=args.data_path_eval, split_file=args.filenames_file_eval, transform=preprocessing_transforms)

            # elif args.dataset_eval == "nyudmt_reverse":
            #     self.testing_samples = NYUDMTR(root=args.data_path_eval, split_file=args.filenames_file_eval, transform=preprocessing_transforms, retname=True,
            #                                     using_ob=args.hypersim_add_contour, is_training=False)     

            # elif args.dataset_eval == "opneworld":
            #     self.testing_samples = OpneWorld(root=args.data_path_eval, split_file=args.filenames_file_eval, transform=preprocessing_transforms, retname=True,
            #                                     )

            elif args.dataset_eval in ["diode_indoor", "diode_outdoor"]:
                self.testing_samples = DIODE_DA(root=args.data_path_eval, split_file=args.filenames_file_eval, transform=preprocessing_transforms, retname=True,
                                                )

            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
