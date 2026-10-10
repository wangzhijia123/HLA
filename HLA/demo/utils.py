"""HLA inference wrapper using a locally supplied full HLA checkpoint."""
import torch
from torch import nn
from src.hla.matcher import HLA as HLAMatcher
from src.hla.backbone import ConvNeXtV2Backbone
from src.hla.checkpoints import load_hla_state_dict


class HLA(nn.Module):
    def __init__(self, checkpoint_path, config):
        super().__init__()
        self.backbone = ConvNeXtV2Backbone()
        self.matcher = HLAMatcher(config)
        self.load_state_dict(load_hla_state_dict(checkpoint_path), strict=True)

    def forward(self, data):
        self.backbone(data)
        return self.matcher(data)


cfg = {
    'coarse': {'d_model': 256},
    'fine': {'d_model': 64, 'dsmax_temperature': 0.1,
             'thr': 0.1, 'inference': True},
    'match_coarse': {'thr': 0.2, 'use_sm': True,
                     'border_rm': 2, 'dsmax_temperature': 0.1,
                     'inference': True},
    'fine_window_size': 5,
    'resolution': [8, 2],
}
