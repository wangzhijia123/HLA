"""ConvNeXt V2 Nano pyramid backbone used by HLA.

Use HLA_BACKBONE_WEIGHTS for the compatible pretrained initialization used
for training. Evaluation instead loads the full HLA checkpoint.
"""
import os
from pathlib import Path
import torch
from torch import nn
from einops import rearrange
from kornia.utils import create_meshgrid
from src.convnextv2.convnextv2 import convnextv2_nano

class ConvNeXtV2Backbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn = convnextv2_nano()
        self.cnn.norm = None
        self.cnn.head = None
        self.cnn.downsample_layers[2] = None
        self.cnn.downsample_layers[3] = None
        self.cnn.stages[2] = None
        self.cnn.stages[3] = None
        checkpoint_path = os.environ.get('HLA_BACKBONE_WEIGHTS')
        if checkpoint_path:
            weights_file = Path(checkpoint_path).expanduser()
            if not weights_file.is_file():
                raise FileNotFoundError(f'Backbone checkpoint not found: {weights_file}')
            weights = torch.load(weights_file, map_location='cpu', weights_only=True)
            if isinstance(weights, dict) and 'state_dict' in weights:
                weights = weights['state_dict']
            self.cnn.load_state_dict(weights, strict=True)
        else:
            # A full HLA checkpoint overrides these initial parameters during evaluation.
            # To reproduce training, set HLA_BACKBONE_WEIGHTS to compatible pretrained weights.
            pass
        self.lin_4 = nn.Conv2d(80, 128, 1)
        self.lin_8 = nn.Conv2d(160, 256, 1)

    def forward(self, data):
        B, _, H, W = data['imagec_0'].shape
        x = torch.cat([data['imagec_0'], data['imagec_1']], 0)
        feature_pyramid = self.cnn.forward_features_8(x)
        feat_8_0, feat_8_1 = self.lin_8(feature_pyramid[8]).split(B)
        feat_4_0, feat_4_1 = self.lin_4(feature_pyramid[4]).split(B)
        scale = 8
        h_8, w_8 = (H // scale, W // scale)
        device = data['imagec_0'].device
        grid = [rearrange((create_meshgrid(h_8, w_8, False, device) * scale).squeeze(0), 'h w t->(h w) t')] * B
        grid_8 = torch.stack(grid, 0)
        data.update({'bs': B, 'c': feat_8_0.shape[1], 'h_8': h_8, 'w_8': w_8, 'hw_8': h_8 * w_8, 'feat_8_0': feat_8_0, 'feat_8_1': feat_8_1, 'feat_4_0': feat_4_0, 'feat_4_1': feat_4_1, 'grid_8': grid_8})
