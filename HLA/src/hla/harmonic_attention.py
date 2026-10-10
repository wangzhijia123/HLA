"""HLA: harmonic attention and signed four-branch Preservation Feature Map.

The default configuration retains the original experimental operations; alternative
feature-map modes are provided for ablation studies via HLA_PFM_MODE.

This file was refactored to remove inactive experiment copies. For lineage and
copyright notices, see THIRD_PARTY_NOTICES.md in the repository root.
"""

from contextlib import nullcontext
import os
import torch
from torch import nn
import torch.nn.functional as F
from src.utils.profiler import PassThroughProfiler
from src.utils.review_diagnostics import add_review_flops, add_review_macs

class PreservationFeatureMap(nn.Module):

    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.review_timer = None

    def forward(self, x_complex, energy):
        timer = self.review_timer
        measure = timer.measure if timer is not None else lambda _name: nullcontext()
        n_complex = x_complex.numel()
        add_review_flops('hla.pfm_branch_scale', 2 * n_complex)
        add_review_flops('hla.pfm_sign_inversion', 2 * n_complex)
        add_review_flops('hla.pfm_four_softplus_branches', 12 * n_complex)
        with measure('hla_pfm_four_branches'):
            real = x_complex.real * energy
            imag = x_complex.imag * energy
            real_pos = F.softplus(real)
            real_neg = F.softplus(-real)
            imag_pos = F.softplus(imag)
            imag_neg = F.softplus(-imag)
        with measure('hla_pfm_concatenation'):
            return torch.cat([real_pos, real_neg, imag_pos, imag_neg], dim=-1)


class HarmonicLinearAttention(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, order=2, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads or self.head_dim % 2:
            raise ValueError('dim must be divisible by num_heads with an even head dimension')
        if not isinstance(order, int) or order < 1:
            raise ValueError('Harmonic order must be a positive integer')
        self.order = order
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.feature_map = PreservationFeatureMap(self.head_dim)
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.attn_eps = 1e-06
        self.review_timer = None
        self.pfm_mode = os.environ.get('HLA_PFM_MODE', 'full').strip().lower()
        valid_modes = {'full', 'raw_ri', 'signed4', 'relu4', 'elu4', 'amplitude4', 'phase_only'}
        if self.pfm_mode not in valid_modes:
            raise ValueError(f'Unknown HLA_PFM_MODE={self.pfm_mode!r}; expected one of {sorted(valid_modes)}')
        self.pfm_diagnostics = None
        self.pfm_label = f'order{order}'

    def forward(self, data, x, source, x_mask=None, source_mask=None):
        B, N, C = x.shape
        H = data['h_8']
        W = data['w_8']
        if source.shape[1] != N or H * W != N:
            raise ValueError('HLA currently requires equal-length coarse token grids; '
                             'resize/pad both images to the same resolution.')
        h_dim = self.head_dim
        timer = self.review_timer
        measure = timer.measure if timer is not None else lambda _name: nullcontext()
        self.feature_map.review_timer = timer
        monitor = self.pfm_diagnostics if self.training else None
        entry = monitor.begin_attention(self.pfm_label) if monitor is not None else None
        with measure('hla_qkv_projection'):
            q = self.q(x).view(B, N, self.num_heads, h_dim).permute(0, 2, 1, 3)
            k = self.k(source).view(B, N, self.num_heads, h_dim).permute(0, 2, 1, 3)
            v = self.v(source).view(B, N, self.num_heads, h_dim).permute(0, 2, 1, 3)
        if x_mask is not None:
            q = q * x_mask.view(B, 1, N, 1)
            add_review_flops('hla.mask_multiply', q.numel())
        if source_mask is not None:
            k = k * source_mask.view(B, 1, N, 1)
            v = v * source_mask.view(B, 1, N, 1)
            add_review_flops('hla.mask_multiply', k.numel() + v.numel())

        def get_phi(t, label):
            with measure('hla_complex_representation_and_normalization'):
                t_complex = torch.view_as_complex(t.reshape(B, self.num_heads, N, h_dim // 2, 2))
                n_complex = t_complex.numel()
                add_review_flops('hla.complex_magnitude', 4 * n_complex)
                r = t_complex.abs()
                if entry is not None:
                    monitor.tensor(entry, f'{label}_magnitude', r, near_zero=self.attn_eps)
                energy = F.softplus(r)
                add_review_flops('hla.harmonic_energy_softplus', 3 * n_complex)
                norm_z = t_complex / (r + self.attn_eps)
                if entry is not None:
                    monitor.tensor(entry, f'{label}_normalized_complex', norm_z)
                add_review_flops('hla.complex_norm_epsilon_add', n_complex)
                add_review_flops('hla.complex_normalization', 2 * n_complex)
            with measure('hla_harmonic_power'):
                harmonic_z = torch.pow(norm_z, self.order)
                if entry is not None:
                    monitor.tensor(entry, f'{label}_harmonic', harmonic_z)
                add_review_flops('hla.integer_harmonic_power', 6 * max(int(self.order) - 1, 0) * n_complex)
            if self.pfm_mode == 'full':
                phi = self.feature_map(harmonic_z, energy)
            elif self.pfm_mode == 'phase_only':
                phi = self.feature_map(harmonic_z, torch.ones_like(energy))
            elif self.pfm_mode == 'amplitude4':
                phi = torch.cat([energy, energy, energy, energy], dim=-1)
            else:
                real = harmonic_z.real * energy
                imag = harmonic_z.imag * energy
                if self.pfm_mode == 'raw_ri':
                    phi = torch.view_as_real(harmonic_z * energy).reshape(B, self.num_heads, N, h_dim)
                else:
                    signed = torch.cat([real, -real, imag, -imag], dim=-1)
                    if self.pfm_mode == 'signed4':
                        phi = signed
                    elif self.pfm_mode == 'relu4':
                        phi = F.relu(signed)
                    else:
                        phi = F.elu(signed) + 1.0
            if entry is not None:
                monitor.tensor(entry, f'phi_{label}', phi)
            return phi
        phi_q = get_phi(q, 'q')
        phi_k = get_phi(k, 'k')
        if entry is not None:
            with torch.no_grad():
                for label, mask, phi in (('q', x_mask, phi_q), ('k', source_mask, phi_k)):
                    if mask is None:
                        continue
                    masked = ~mask.reshape(B, 1, N, 1).bool()
                    entry[f'{label}_masked_fraction'] = float(masked.float().mean().item())
                    if bool(masked.any().item()):
                        masked_values = phi.detach().float().masked_select(masked.expand_as(phi))
                        valid_values = masked_values[torch.isfinite(masked_values)]
                        entry[f'{label}_masked_phi_mean'] = float(valid_values.mean().item()) if valid_values.numel() else None
        phi_dim = phi_q.shape[-1]
        head_dim = v.shape[-1]
        add_review_macs('hla.linear_attention_kv', B * self.num_heads * phi_dim * N * head_dim)
        add_review_macs('hla.linear_attention_query_kv', B * self.num_heads * N * phi_dim * head_dim)
        add_review_macs('hla.linear_attention_denominator', B * self.num_heads * N * phi_dim)
        add_review_flops('hla.linear_attention_key_reduction', B * self.num_heads * max(N - 1, 0) * phi_dim)
        add_review_flops('hla.linear_attention_normalize', B * self.num_heads * N * head_dim)
        with measure('hla_linear_attention_aggregation'):
            kv = torch.matmul(phi_k.transpose(-2, -1), v)
            if entry is not None:
                monitor.tensor(entry, 'kv', kv)
            num = torch.matmul(phi_q, kv)
            if entry is not None:
                monitor.tensor(entry, 'num', num)
            k_sum = phi_k.sum(dim=2, keepdim=True)
            den_raw = torch.matmul(phi_q, k_sum.transpose(-2, -1))
            if entry is not None:
                monitor.denominator(entry, den_raw, self.attn_eps)
            den = den_raw.clamp_min(self.attn_eps)
            out = (num / den).transpose(1, 2).reshape(B, N, C)
            if entry is not None:
                monitor.tensor(entry, 'attention_output', out)
        with measure('hla_lepe_and_output_add'):
            v_img = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            lepe_res = self.lepe(v_img).flatten(2).transpose(1, 2)
            add_review_flops('hla.output_residual_add', out.numel())
            return out + lepe_res


class HLABlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=2.0, order=2, qkv_bias=True, drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = HarmonicLinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias, order=order)
        self.out_proj = nn.Linear(dim, dim)
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, data, x, source, x_mask=None, source_mask=None):
        H = data['h_8']
        W = data['w_8']
        B, L, C = x.shape
        x1 = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.cpe1(x1.contiguous()).flatten(2).permute(0, 2, 1)
        source1 = source.reshape(B, H, W, C).permute(0, 3, 1, 2)
        source = source + self.cpe1(source1.contiguous()).flatten(2).permute(0, 2, 1)
        add_review_flops('mlla.pointwise_residuals_and_gating', 6 * B * L * C)
        shortcut = x
        x = self.norm1(x)
        source = self.norm1(source)
        act_res = self.act(self.act_proj(x))
        x_proj = self.in_proj(x).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.act(self.dwc(x_proj)).flatten(2).permute(0, 2, 1)
        src_proj = self.in_proj(source).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        source = self.act(self.dwc(src_proj)).flatten(2).permute(0, 2, 1)
        x = self.attn(data, x, source, x_mask, source_mask)
        x = self.out_proj(x * act_res)
        x = shortcut + x
        x2 = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.cpe2(x2.contiguous()).flatten(2).permute(0, 2, 1)
        x = x + self.mlp(self.norm2(x))
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HLAInteractionStack(nn.Module):

    def __init__(self, feature_dim, input_resolution, nhead, mlp_ratio=2.0):
        super().__init__()
        self.layer_names = ['self', 'cross'] * 2
        self.layers = nn.ModuleList()
        for name in self.layer_names:
            order = 1 if name == 'self' else 3
            self.layers.append(HLABlock(feature_dim, input_resolution, nhead, mlp_ratio, order=order))

    def forward(self, data, feat0, feat1, mask0=None, mask1=None):
        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if name == 'self':
                feat0 = layer(data, feat0, feat0, mask0, mask0)
                feat1 = layer(data, feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(data, feat0, feat1, mask0, mask1)
                feat1 = layer(data, feat1, feat0, mask1, mask0)
        return (feat0, feat1)


class HLAFeatureInteraction(nn.Module):

    def __init__(self, feature_dim: int, profiler=None):
        super().__init__()
        self.profiler = profiler or PassThroughProfiler()
        self.review_timer = None
        self.nhead = 8
        dim = 256
        input_resolution = [104, 104]
        self.atten = HLAInteractionStack(dim, input_resolution, self.nhead, mlp_ratio=2.0)

    def forward(self, data):
        feat_8_0, feat_8_1 = (data['feat_8_0'], data['feat_8_1'])
        mask_c0 = mask_c1 = None
        if 'mask0' in data:
            mask_c0, mask_c1 = (data['mask0'], data['mask1'])
        if self.review_timer is not None:
            for block in self.atten.layers:
                block.attn.review_timer = self.review_timer
        feat_8_0, feat_8_1 = self.atten(data, feat_8_0.permute(0, 2, 1).contiguous(), feat_8_1.permute(0, 2, 1).contiguous(), mask_c0, mask_c1)
        data.update({'feat_8_0': feat_8_0, 'feat_8_1': feat_8_1})
