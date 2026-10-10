"""Synthetic CPU pass through the complete coarse-to-fine matcher.
No backbone data or trained weights are needed. This is not a benchmark test.
"""
import torch
from src.hla.matcher import HLA


def test_synthetic_coarse_to_fine_matcher():
    torch.manual_seed(42)
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
    model = HLA(cfg).eval()
    h = w = 8
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid = torch.stack([xs, ys], dim=-1).reshape(1, -1, 2).float() * 8
    data = {
        'imagec_0': torch.rand(1, 3, 64, 64),
        'imagec_1': torch.rand(1, 3, 64, 64),
        'feat_8_0': torch.randn(1, 256, 8, 8),
        'feat_8_1': torch.randn(1, 256, 8, 8),
        'feat_4_0': torch.randn(1, 128, 16, 16),
        'feat_4_1': torch.randn(1, 128, 16, 16),
        'grid_8': grid,
        'bs': 1, 'c': 256, 'h_8': h, 'w_8': w, 'hw_8': h*w,
    }
    with torch.inference_mode():
        model(data)
    assert data['mkpts0_f'].shape[-1] == 2
    assert data['mkpts1_f'].shape[-1] == 2
    assert torch.isfinite(data['mkpts0_f']).all()
    assert torch.isfinite(data['mkpts1_f']).all()
