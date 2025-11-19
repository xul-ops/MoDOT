from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):

        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result) 
        avg_out=self.se(avg_result) 
        output=self.sigmoid(max_out+avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        # x:(B,C,H,W)
        max_result,_=torch.max(x,dim=1,keepdim=True)  
        avg_result=torch.mean(x,dim=1,keepdim=True)   
        result=torch.cat([max_result,avg_result],1)   
        output=self.conv(result)                     
        output=self.sigmoid(output)                   
        return output


class ConvBnRelu(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, bias=False, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBnRelu, self).__init__(OrderedDict([
            ('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                     padding=padding, groups=groups, bias=bias)),
            ('bn', torch.nn.BatchNorm2d(out_planes)),
            ('relu', torch.nn.ReLU(inplace=True))
        ]))
        

class ConvBnRelu2(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, padding=1, bias=False, stride=1, groups=1):
        super(ConvBnRelu2, self).__init__(OrderedDict([
            ('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                     padding=padding, groups=groups, bias=bias)),
            ('bn', torch.nn.BatchNorm2d(out_planes)),
            ('relu', torch.nn.ReLU(inplace=True))
        ]))


def crop(data1, data2):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    if h1 == h2 and w1 == w2:
        return data1
    if h1 < h2 or w1 < w2:
        pad_h = (h2 - h1) // 2 + 1
        pad_h = pad_h if pad_h > 0 else 0
        pad_w = (w2 - w1) // 2 + 1
        pad_w = pad_w if pad_w > 0 else 0
        data1 = torch.nn.ConstantPad2d((pad_w, pad_w, pad_h, pad_h), 0)(data1)
        _, _, h1, w1 = data1.size()
    assert (h2 <= h1 and w2 <= w1)
    offset_h = (h1 - h2) // 2
    offset_w = (w1 - w2) // 2
    data = data1[:, :, offset_h:offset_h + h2, offset_w:offset_w + w2]
    return data


class BasicDecoder(nn.Module):
    def __init__(self,
                 feature_channels=[512, 256, 128, 64, 32],
                 half_feature_channels=True,
                 feature_scales=[16, 8, 4, 2, 1],
                 path_name='boundary',
                 num_end_out=1,
                 use_pixel_shuffle=False,
                 lite=False):
        '''
        path_name = 'boundary' or 'ori'
        '''
        super(BasicDecoder, self).__init__()
        assert len(feature_channels) == len(feature_scales)

        self.half_feature_channels = half_feature_channels
        self.path_name = path_name

        if self.half_feature_channels:
            self.half_features_op = nn.ModuleList()
            for num_channel in feature_channels:
                self.half_features_op.append(
                    ConvBnRelu(num_channel, num_channel // 2, 1)
                )
            feature_channels = [num_channel //
                                2 for num_channel in feature_channels]

        self.decoder_fuses = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        self.out_upsamples = nn.ModuleList()

        for idx, num_channel in enumerate(feature_channels):
            in_num_channel = num_channel if idx == 0 else num_channel * 2

            # decoder
            self.decoder_fuses.append(nn.Sequential(
                ConvBnRelu(in_num_channel, in_num_channel, 3),
                ConvBnRelu(in_num_channel, in_num_channel, 3)
            ))

            if idx + 1 < len(feature_channels):
                if path_name == 'boundary':
                    self.out_upsamples.append(nn.Sequential(
                        self.make_upsample(in_num_channel, 1, feature_scales[idx],
                                           pixel_shuffle=False)
                    ))
                self.decoder_upsamples.append(nn.Sequential(
                    self.make_upsample(in_num_channel,
                                       feature_channels[idx + 1],
                                       feature_scales[idx] // feature_scales[idx+1],
                                       pixel_shuffle=use_pixel_shuffle)
                ))

        if path_name == 'boundary':
            end_channel_num = 8 if lite else 64
            self.end = nn.Sequential(
                ConvBnRelu(feature_channels[-1] * 2, end_channel_num, 3),
                nn.Conv2d(end_channel_num, num_end_out, 1, padding=0, bias=False)
            )
        elif path_name == 'ori':
            end_channel_num = 8 if lite else 32
            self.end = nn.Sequential(
                ConvBnRelu(feature_channels[-1] * 2, end_channel_num, 1),
                ConvBnRelu(end_channel_num, end_channel_num, 3),
                nn.Conv2d(end_channel_num, num_end_out, 3, padding=1)
            )
        else:
            raise AttributeError(f"path_name is restricted among in ['boundary', 'ori'], got {path_name} instead!")

    def make_upsample(self, in_planes, out_planes, scale_factor=2, pixel_shuffle=False):
        if scale_factor == 1:
            return nn.Conv2d(in_planes, out_planes, 1, padding=0, bias=False)

        if pixel_shuffle:
            out_planes = out_planes * scale_factor ** 2
        layers = OrderedDict()
        if in_planes != out_planes or pixel_shuffle:
            layers['conv'] = ConvBnRelu(in_planes, out_planes, 1)

        if pixel_shuffle:
            layers['upsample'] = nn.PixelShuffle(scale_factor)
        else:
            layers['upsample'] = nn.Upsample(
                scale_factor=scale_factor, mode='bilinear', align_corners=True)

        return nn.Sequential(layers)

    def forward(self, features):
        # print(features[1].shape)
        # print(len(features), len(self.decoder_fuses))
        assert len(features) == len(self.decoder_fuses)

        if self.half_feature_channels:
            features = [half_feat_op(feat) for half_feat_op, feat in zip(
                self.half_features_op, features)]

        outs = []

        decoder_x = None
        for idx, feature in enumerate(features):
            if idx == 0:
                decoder_x = feature
            else:
                decoder_x = crop(decoder_x, feature)
                decoder_x = torch.cat([feature, decoder_x], 1)

            decoder_x = self.decoder_fuses[idx](decoder_x)

            if idx < len(self.decoder_upsamples):
                if self.path_name == 'boundary':
                    outs.append(self.out_upsamples[idx](decoder_x))

                decoder_x = self.decoder_upsamples[idx](decoder_x)

        outs.append(self.end(decoder_x))
        return outs


class BasicDecoder2(nn.Module):
    def __init__(self,
                 feature_channels=[512, 256, 128, 64, 32],
                 half_feature_channels=True,
                 feature_scales=[16, 8, 4, 2, 1],
                 path_name='boundary',
                 num_end_out=1,
                 use_pixel_shuffle=False,
                 lite=False):
        '''
        path_name = 'boundary' or 'ori'
        '''
        super(BasicDecoder2, self).__init__()
        assert len(feature_channels) == len(feature_scales)

        self.half_feature_channels = half_feature_channels
        self.path_name = path_name

        if self.half_feature_channels:
            self.half_features_op = nn.ModuleList()
            for num_channel in feature_channels:
                self.half_features_op.append(
                    ConvBnRelu(num_channel, num_channel // 2, 1)
                )
            feature_channels = [num_channel //
                                2 for num_channel in feature_channels]

        self.decoder_fuses = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        self.out_upsamples = nn.ModuleList()

        for idx, num_channel in enumerate(feature_channels):
            in_num_channel = num_channel if idx == 0 else num_channel * 2

            # decoder
            self.decoder_fuses.append(nn.Sequential(
                ConvBnRelu(in_num_channel, in_num_channel, 3),
                ConvBnRelu(in_num_channel, in_num_channel, 3)
            ))

            if idx + 1 < len(feature_channels):
                if path_name == 'boundary':
                    self.out_upsamples.append(nn.Sequential(
                        self.make_upsample(in_num_channel, 1, feature_scales[idx],
                                           pixel_shuffle=False)
                    ))
                self.decoder_upsamples.append(nn.Sequential(
                    self.make_upsample(in_num_channel,
                                       feature_channels[idx + 1],
                                       feature_scales[idx] // feature_scales[idx+1],
                                       pixel_shuffle=use_pixel_shuffle)
                ))

        if path_name == 'boundary':
            end_channel_num = 8 if lite else 64
            self.end = nn.Sequential(
                ConvBnRelu(feature_channels[-1] * 2, end_channel_num, 3),
                nn.Conv2d(end_channel_num, num_end_out, 1, padding=0, bias=False)
            )
        elif path_name == 'ori':
            end_channel_num = 8 if lite else 32
            self.end = nn.Sequential(
                ConvBnRelu(feature_channels[-1] * 2, end_channel_num, 1),
                ConvBnRelu(end_channel_num, end_channel_num, 3),
                nn.Conv2d(end_channel_num, num_end_out, 3, padding=1)
            )
        else:
            raise AttributeError(f"path_name is restricted among in ['boundary', 'ori'], got {path_name} instead!")

    def make_upsample(self, in_planes, out_planes, scale_factor=2, pixel_shuffle=False):
        if scale_factor == 1:
            return nn.Conv2d(in_planes, out_planes, 1, padding=0, bias=False)

        if pixel_shuffle:
            out_planes = out_planes * scale_factor ** 2
        layers = OrderedDict()
        if in_planes != out_planes or pixel_shuffle:
            layers['conv'] = ConvBnRelu(in_planes, out_planes, 1)

        if pixel_shuffle:
            layers['upsample'] = nn.PixelShuffle(scale_factor)
        else:
            layers['upsample'] = nn.Upsample(
                scale_factor=scale_factor, mode='bilinear', align_corners=True)

        return nn.Sequential(layers)

    def forward(self, features):
        # print(features[1].shape)
        # print(len(features), len(self.decoder_fuses))
        assert len(features) == len(self.decoder_fuses)

        if self.half_feature_channels:
            features = [half_feat_op(feat) for half_feat_op, feat in zip(
                self.half_features_op, features)]

        outs = []

        decoder_x = None
        for idx, feature in enumerate(features):
            if idx == 0:
                decoder_x = feature
            else:
                decoder_x = crop(decoder_x, feature)
                decoder_x = torch.cat([feature, decoder_x], 1)

            decoder_x = self.decoder_fuses[idx](decoder_x)

            ob_feats = decoder_x

            if idx < len(self.decoder_upsamples):
                if self.path_name == 'boundary':
                    outs.append(self.out_upsamples[idx](decoder_x))

                decoder_x = self.decoder_upsamples[idx](decoder_x)

        outs.append(self.end(decoder_x))
        return outs, ob_feats


class CASM(nn.Module):

    """
    main part of CASM

    """

    def __init__(self, channel=512,reduction=16,kernel_size=5):
        super().__init__()

        self.ca_ob = ChannelAttention(channel=channel,reduction=reduction)
        self.ca_depth = ChannelAttention(channel=channel,reduction=reduction)

        self.sa = MSS_Fuse(channel, kernel_size1=1, kernel_size2=7, kernel_size3=11)

        self.fuse_layer = nn.Conv2d(channel*2, channel, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, ob_feats, depth_feats):
        obf = ob_feats  # (B,C,H,W)
        depf = depth_feats # (B,C,H,W)

        depf_ca_weights = self.ca_depth(depf)
        obf_ca_weights = self.ca_ob(obf)

        fused = self.fuse_layer(torch.concat([obf, depf], dim=1))

        # depf_ca = depf * obf_ca_weights
        # obf_ca =  obf * depf_ca_weights

        depth_out = self.sa(fused) + depf * obf_ca_weights + depf

        ob_out = obf + obf * depf_ca_weights

        return ob_out, depth_out


class MSS_Fuse(torch.nn.Module):
    # 1, 5, 7 and 1, 3, 7
    def __init__(self, planes, kernel_size1=1, kernel_size2=5, kernel_size3=7):
        super().__init__()
      
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(planes, planes, (kernel_size1, kernel_size2), padding=(
                (kernel_size1 - 1) // 2, (kernel_size2 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, (kernel_size1, kernel_size2), padding=(
                (kernel_size1 - 1) // 2, (kernel_size2 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes)
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(planes, planes, (kernel_size2, kernel_size1), padding=(
                (kernel_size2 - 1) // 2, (kernel_size1 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, (kernel_size2, kernel_size1), padding=(
                (kernel_size2 - 1) // 2, (kernel_size1 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes)
        )
        
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(planes, planes, 3, padding=1, bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(planes, planes, (kernel_size1, kernel_size3), padding=(
                (kernel_size1 - 1) // 2, (kernel_size3 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, (kernel_size1, kernel_size3), padding=(
                (kernel_size1 - 1) // 2, (kernel_size3 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes)
        )
        self.conv5 = torch.nn.Sequential(
            nn.Conv2d(planes, planes, (kernel_size3, kernel_size1), padding=(
                (kernel_size3 - 1) // 2, (kernel_size1 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, (kernel_size3, kernel_size1), padding=(
                (kernel_size3 - 1) // 2, (kernel_size1 - 1) // 2), bias=True),
            nn.BatchNorm2d(planes)
        )      
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_end = ConvBnRelu(planes * 3, planes, 1)

    def forward(self, a):
        x1 = self.relu(self.conv1(a) + self.conv2(a))
        x2 = self.conv3(a)
        x3 = self.relu2(self.conv4(a) + self.conv5(a))
        x = self.conv_end(torch.cat([x1, x2, x3], 1))    
        return x

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_b(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out
        
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv3x3_b(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

