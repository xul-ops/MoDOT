import os
import pdb
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from .basic_components import *
from .MoDOT import *

########################################################################################################################

class MoDOT_SSR(nn.Module):

    def __init__(self, args, version=None, max_depth=10.0, checkpoint_path="", **kwargs):
        super().__init__()

        self.max_depth = max_depth
        init_model = MoDOT(args, version=version, max_depth=max_depth, pretrained=None, return_feats=True)
        self.init_model = torch.nn.DataParallel(init_model)

        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.init_model.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}'".format(checkpoint_path))
            del checkpoint
        else:
            print("Didn't load first stage checkpoint")

        for param in self.init_model.parameters():
            param.requires_grad = False  # Ensure freezing

        self.double_ob = torch.nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=1, stride=2),
            )

        # Task-specific heads for final prediction
        heads = {}
        for task in ["depth"]: # ["depth", "ob"]:
            bottleneck1 = Bottleneck(64, 64//4, downsample=None)
            # bottleneck2 = Bottleneck(128, 128//4, downsample=None)
            conv_out_ = nn.Conv2d(64, 1, 1)
            heads[task] = nn.Sequential(bottleneck1, conv_out_)

        self.ob_bottleneck = Bottleneck(64, 64//4, downsample=None)
        self.ob_conv_out_ = nn.Conv2d(64, 1, 1)
        self.conv_down = nn.Conv2d(128, 64, 1)

        self.heads = nn.ModuleDict(heads)

        # self.fuse_ob = conv1x1(2, 1)
        # self.fuse_depth = conv1x1(2, 1)
        self.sigmoid_depth = nn.Sigmoid()

        self.feats_cross_final = CASM(64) 
        print()
        print("Two stage refiner")
        print()
    
    def forward(self, imgs):


        w, h  = imgs.shape[2:]
        out = dict()    

        with torch.no_grad():  # Freeze feature extractor
            d1, ob, e0, ob_feats = self.init_model(imgs)


        ob_feats_up = self.double_ob(ob_feats)
        e00 =  self.conv_down(e0) # nn.PixelShuffle(2)(e0)
        of, df = self.feats_cross_final(ob_feats_up, e00)

        out["depth"] = e00 + df
        c_t = self.heads["depth"](out["depth"])
        out["depth"] = F.interpolate(c_t, (w, h), mode='bilinear')
        out["depth"] = self.sigmoid_depth(out["depth"]) * self.max_depth

        # change ob_feats_up to depth_related
        t = crop(upsample(of, scale_factor=4), ob_feats)  + ob_feats
        ob_refine = self.ob_bottleneck(t)
        ob_final = self.ob_conv_out_(ob_refine)


        ob_final = crop(ob_final, imgs)
        output = [out["depth"], ob_final]

        return output
        

class DispHead(nn.Module):
    def __init__(self, input_dim=100, output_dim=1):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


class DispUnpack(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128):
        super(DispUnpack, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x, output_size):
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x)) # [b, 16, h/4, w/4]
        # x = torch.reshape(x, [x.shape[0], 1, x.shape[2]*4, x.shape[3]*4])
        x = self.pixel_shuffle(x)

        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

