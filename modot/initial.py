import os
import pdb
import torch
from torchvision import transforms
import math

import numpy as np
from losses.loss_schemes import DepthOBLoss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler


class WarmUpLRScheduler(_LRScheduler):

    def __init__(self, optimizer, T_warmup, after_scheduler_name, **kwargs):
        self.T_warmup = T_warmup
        self.after_scheduler = getattr(lr_scheduler, after_scheduler_name)(optimizer=optimizer, **kwargs)
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.T_warmup:
            return self.after_scheduler.get_lr()

        return [base_lr * (self.last_epoch + 1.0) / (self.T_warmup + 1.0) for base_lr in self.base_lrs]

    def get_last_lr(self):
        if self.last_epoch >= self.T_warmup:
            self.after_scheduler.get_last_lr()

        return super(WarmUpLRScheduler, self).get_last_lr()

    def print_lr(self, is_verbose, group, lr, epoch=None):
        if self.last_epoch >= self.T_warmup:
            self.after_scheduler.print_lr(is_verbose, group, lr, epoch)

        return super(WarmUpLRScheduler, self).print_lr(is_verbose, group, lr, epoch)

    def step(self, epoch=None):
        if self.last_epoch >= self.T_warmup:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.T_warmup)
        else:
            return super(WarmUpLRScheduler, self).step(epoch)



def load_checkpoint(args, model, optimizer):
    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                    best_eval_steps = checkpoint['best_eval_steps']
                except KeyError:
                    print("Could not load values for online evaluation")

            print("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
        del checkpoint
    
    return model, optimizer, model_just_loaded


def init_criterion(args):
    return DepthOBLoss(args)


def init_optimizer(args, model):
    optimizer = torch.optim.Adam([{'params': model.module.parameters()}],
                                lr=args.learning_rate)
    return optimizer


def init_scheduler(cfg, optimizer):
    print(' => scheduler')
    print(f'  name: {cfg.scheduler_name}')
    print(f'  scheduler_mode: {cfg.scheduler_mode}')
    print(f'  warmup_epochs: {cfg.warmup_epochs}')

    scheduler_param = eval(cfg.scheduler_param)
    print(f'  scheduler_param: {scheduler_param}')

    T_scale = 1
    if cfg.scheduler_mode == 'epoch':
        T_scale = 1
    elif cfg.scheduler_mode == 'iter':
        T_scale = len(ENV.data_loaders['train'])
    else:
        raise ValueError(f'cfg.scheduler_mode={cfg.scheduler_mode} is not supported')

    T_warmup = cfg.warmup_epochs * T_scale
    if cfg.scheduler_name == 'CosineAnnealingLR':
        if 'T_max' not in scheduler_param.keys():
            scheduler_param['T_max'] = cfg.epoch * T_scale

    scheduler = WarmUpLRScheduler(optimizer, T_warmup,
                                  after_scheduler_name=cfg.scheduler_name, **scheduler_param)
    return scheduler


# some MT models
def get_head(p, backbone_channels, task):
    """ Return the decoder head """

    if p['head'] == 'mlp':
        from nnmt.transformers.transformer_decoder import MLPHead
        return MLPHead(backbone_channels, 1)

    elif p['head'] == 'deeplab':
        from nnmt.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, 1)

    elif p['head'] == 'hrnet':
        from nnmt.seg_hrnet import HighResolutionHead
        return HighResolutionHead(backbone_channels, 1)

    else:
        raise NotImplementedError


def init_model(args):

    print(f"=> Model: {args.model_name}")

    args.backbone_name = args.encoder
    if args.mode == "eval":
        args.pretrain = None

    if args.model_name == "MoDOT_SSR":
        from networks.MoDOT_SSR import MoDOT_SSR
        model = MoDOT_SSR(args, version=args.encoder, max_depth=args.max_depth, checkpoint_path=args.ts_checkpoint_path)
 
    elif args.model_name == "MoDOT":
        from networks.MoDOT import MoDOT
        model = MoDOT(args, version=args.encoder, max_depth=args.max_depth, pretrained=args.pretrain)

    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    return model



def init_transformations(args, is_train=True):
    """ Return transformations for training and evaluationg from MTI-Net """
    from dataloaders import transforms
    import torchvision


    # Training transformations
    if args.dataset == 'nyudmt' :
        train_transforms = torchvision.transforms.Compose([ # from ATRC
            # transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            # transforms.RandomCrop(size=(448, 576), cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.PadImage(size=(448, 576)),
            # transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

    else:
        train_transforms = torchvision.transforms.Compose([ 
            # transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=(args.input_width, args.input_height), cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor(),
        ])

    # Testing 
    valid_transforms = torchvision.transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.PadImage(size=(480, 640)),
        # transforms.AddIgnoreRegions(),
        transforms.ToTensor(),
    ])
    
    if args.is_two_stage:
        del train_transforms
        del valid_transforms

        train_transforms = torchvision.transforms.Compose([ 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor(),
        ])

        # Testing 
        valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor(),
        ])        

    return train_transforms, valid_transforms        


def init_dataloader(args, is_train=True, train_transforms=None, valid_transforms=None):
    
    from dataloaders.dataloader_obd import NewDataLoader

    if is_train:
        dataloader = NewDataLoader(args, 'train', train_transforms)
        dataloader_eval = 0 # NewDataLoader(args, 'online_eval')

        return dataloader, dataloader_eval

    else:
        dataloader = NewDataLoader(args, 'online_eval', valid_transforms)

        return dataloader


def init_saving(args):
    # synocc is OB-FUTURE
    if args.dataset == "synocc":
        start_epoch = 15
        start_iter = 30000
        end_iter = 50000
        if args.is_two_stage:
            start_epoch = 0
            start_iter = 0            
    elif args.dataset in ["nyudmt", "nyudmt_reverse"]:
        start_epoch = 30
        start_iter = 40000
        end_iter = 50000
        if args.is_two_stage:
            start_epoch = 0
            start_iter = 0  
            # end_iter = 60000
    else:
        start_epoch = 10
        start_iter = 30000
        end_iter = 50000
        if args.is_two_stage:
            start_epoch = 0
            start_iter = 6000
            end_iter = 30000
            args.save_freq = 1000 
        
    
    return start_epoch, start_iter, end_iter


def init_eval_saving(args):

    if args.dataset_eval == "synocc":
        start_eval_index = 15
        end_eval_index = 20
        eval_step = 1
        save_interval = 50
        vis_res = False

        if args.is_two_stage:
            start_eval_index = 0
            end_eval_index = 10   

    elif args.dataset_eval == "hypersim":
        start_eval_index = 30000
        end_eval_index = 52000
        eval_step = 2000 # args.save_iter_freq
        save_interval = 50
        vis_res = False

        if args.is_two_stage:
            start_epoch = 0
            start_eval_index = 6000  
            end_eval_index = 31000 # 62000
            eval_step = 1000

    else:
        start_eval_index = 30
        end_eval_index = 51
        eval_step = 1
        save_interval = 1
        vis_res = False

        if args.is_two_stage:
            start_eval_index = 0
            end_eval_index = 51
        
    return start_eval_index, end_eval_index, eval_step, save_interval, vis_res