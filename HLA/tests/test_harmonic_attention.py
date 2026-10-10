"""CPU formula/shape regression tests; these do not validate trained AUC."""
import pytest
import torch
import torch.nn.functional as F

from src.hla.harmonic_attention import PreservationFeatureMap, HarmonicLinearAttention


def test_pfm_signed_branches_and_shape():
    feature_map = PreservationFeatureMap(8)
    z = torch.tensor([[[[1.0 + 2.0j, -2.0 - 1.0j]]]], dtype=torch.complex64)
    energy = torch.tensor([[[[2.0, 0.5]]]], dtype=torch.float32)
    phi = feature_map(z, energy)
    assert phi.shape == (1, 1, 1, 8)
    assert torch.all(phi > 0)
    a = z.real * energy
    b = z.imag * energy
    assert torch.allclose(phi[..., :2] - phi[..., 2:4], a, atol=1e-6)
    assert torch.allclose(phi[..., 4:6] - phi[..., 6:8], b, atol=1e-6)


@pytest.mark.parametrize('order', [1, 3])
def test_harmonic_attention_forward_and_backward(order):
    torch.manual_seed(3)
    layer = HarmonicLinearAttention(dim=32, input_resolution=(4, 4), num_heads=8, order=order)
    data = {'h_8': 4, 'w_8': 4}
    x = torch.randn(2, 16, 32, requires_grad=True)
    result = layer(data, x, x)
    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_invalid_head_dim_rejected():
    with pytest.raises(ValueError):
        HarmonicLinearAttention(dim=24, input_resolution=(2, 2), num_heads=8, order=1)


def test_no_paper_auc_claim_is_inferred_from_random_weights():
    model = HarmonicLinearAttention(dim=32, input_resolution=(2, 2), num_heads=8, order=1)
    assert model.order == 1
