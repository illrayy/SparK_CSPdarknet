# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import List
from timm.models.registry import register_model


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul=1, wid_mul=1, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu",):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.wid_mul = wid_mul
        Conv = DWConv if depthwise else BaseConv


        base_channels   = int(wid_mul * 64)  # 64
        base_depth      = max(round(dep_mul * 3), 1)  # 3
        
        #focus
        #downsample_ratio,ch
        #2, 64
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        
        #4, 128
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        #8, 256
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        #16, 512
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        #32, 1024
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )
        
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    
    def get_downsample_ratio(self) -> int:
        return 32
    
    def get_feature_map_channels(self) -> List[int]:
        return [256, 512, 1024]*self.wid_mul
    
    def forward(self, x, hierarchical=False):
        ls = []
        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3 8, 256
        #-----------------------------------------------#
        x = self.dark3(x)
        ls.append(x)
        #-----------------------------------------------#
        #   dark4 16, 512
        #-----------------------------------------------#
        x = self.dark4(x)
        ls.append(x)
        #-----------------------------------------------#
        #   dark5 32, 1024
        #-----------------------------------------------#
        x = self.dark5(x)
        ls.append(x)
        return ls


@register_model
def CSPDarknet_nano(pretrained=False, **kwargs):
    model = CSPDarknet(0.33, 0.25, depthwise=True)
    return model

def CSPDarknet_tiny(pretrained=False, **kwargs):
    model = CSPDarknet(0.33, 0.375)
    return model

def CSPDarknet_s(pretrained=False, **kwargs):
    model = CSPDarknet(0.33, 0.5)
    return model

def CSPDarknet_m(pretrained=False, **kwargs):
    model = CSPDarknet(0.67, 0.75)
    return model

def CSPDarknet_l(pretrained=False, **kwargs):
    model = CSPDarknet(1, 1)
    return model

def CSPDarknet_x(pretrained=False, **kwargs):
    model = CSPDarknet(1.33, 1.25)
    return model

@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('CSPDarknet_nano')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
