import pdb
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
from .basic_components import *

########################################################################################################################


class MoDOT(nn.Module):

    def __init__(self, args, version=None, pretrained=None, 
                    frozen_stages=-1, max_depth=100.0, return_feats=False, **kwargs):
        super().__init__()

        self.return_feats = return_feats

        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]
        elif version[:-2] == 'small':
            embed_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            in_chans = 3, 
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]

        # depth decoders
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)
        # high-level
        self.decoder = PSP(**decoder_cfg)
        # depth head
        self.max_depth = max_depth
        self.up_mode = 'bilinear'
        self.disp_head = DispHead(input_dim=crf_dims[0])

        # CASM part for OB
        self.skip_feat1 = ConvBnRelu(1536, 256, 1)
        self.skip_feat2 = ConvBnRelu(768, 128, 1)
        self.skip_feat3 = ConvBnRelu(384, 64, 1)
        self.skip_feat4 = ConvBnRelu(192, 128, 1)

        # EIP
        out_shape = [1536, 768, 384, 192]
        num_decoder_end_c = [16, 64]
        bb_feature_scales = [16, 8, 4, 2, 1]
        # Image Path
        self.ob_convolution = torch.nn.Sequential(
            ConvBnRelu(3, num_decoder_end_c[0], 1),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[0], 3),
            ConvBnRelu(num_decoder_end_c[0], num_decoder_end_c[1], 1)
            )
        
        self.ca_img = ChannelAttention(num_decoder_end_c[1], reduction=4) # ECAA(kernel_size=3) 
        self.sa_img = SpatialAttention(kernel_size=5)  # ks = 5
        self.fuse_aimg = ConvBnRelu(num_decoder_end_c[1] * 2, num_decoder_end_c[1], 1)


        # 4 torch.Size([2, 192, 80, 80]) torch.Size([2, 384, 40, 40]) torch.Size([2, 768, 20, 20]) torch.Size([2, 1536, 10, 10])
        FeatsAttn = CASM 
        self.feats_cross1 = FeatsAttn(256)
        self.feats_cross2 = FeatsAttn(128) 
        self.feats_cross3 = FeatsAttn(64)  
        self.feats_cross4 = FeatsAttn(128) 

        # if args.ob_loss_type == "normal" and args.hypersim_b_type == "normal":
        #     t = 3 
        # if args.ob_loss_type == "sem_seg" and args.hypersim_b_type == "sem_seg":
        #     t = 40
        #     print("OB branch output shape: ", t)
        # else:
        #     t = 1
        t = 1
        print("OB branch output shape: ", t)
        self.ob_decoder = BasicDecoder2(
            feature_channels=[256, 128, 64, 128, num_decoder_end_c[1]],
            feature_scales=[16, 8, 4, 4, 1],
            path_name='boundary',
            num_end_out=t,
            use_pixel_shuffle=False,
        )
        
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()


    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 4*H, 4*W)

    def forward(self, imgs):


        feats = self.backbone(imgs)
        ppm_out = self.decoder(feats)

        o3 = self.skip_feat1(upsample( feats[3], 2))  # + e3
        o2 = self.skip_feat2(upsample( feats[2], 2))  # + e2
        o1 = self.skip_feat3(upsample( feats[1], 2))  # + e1
        o0 = self.skip_feat4(feats[0])                # + e0

        e3 = self.crf3(feats[3], ppm_out)
        # PixelShuffle - part of CAMS
        e3 = nn.PixelShuffle(2)(e3)
        o3, e3 = self.feats_cross1(o3, e3)

        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        o2, e2 = self.feats_cross2(o2, e2)

        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        o1, e1 = self.feats_cross3(o1, e1)

        e0 = self.crf0(feats[0], e1)
        o0, e0 = self.feats_cross4(o0, e0)

        d1 = self.disp_head(e0, 4)

        img_cues = self.ob_convolution(imgs)      
        ca_img_cues = self.ca_img(img_cues) * img_cues + img_cues
        sa_img_cues = self.sa_img(img_cues) * img_cues + img_cues
        img_cues = self.fuse_aimg(torch.concat([ca_img_cues, sa_img_cues], dim=1))

        # ob = self.ob_decoder([o3, o2, o1, o0, img_cues])    
        ob, ob_feats = self.ob_decoder([o3, o2, o1, o0, img_cues])  

        ob = [crop(item, imgs) for item in ob]

        d1 = crop(d1, imgs)

        if self.return_feats:
            return [d1 * self.max_depth, ob, e0, ob_feats]
        else:
            return [d1 * self.max_depth, ob]


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



