"""Load full HLA model checkpoints, including earlier locally trained checkpoints.

Only load checkpoints obtained from a trusted source; PyTorch serialization can
execute code on some versions / with non-weight metadata.
"""
from pathlib import Path
import torch


def load_hla_state_dict(checkpoint_path):
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'HLA checkpoint not found: {path}')
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch versions without weights_only
        checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError('Checkpoint must contain a model state_dict or a state_dict entry')
    normalized = {}
    for key, value in state.items():
        # Match parameter names prior to the public HLA namespace refactor.
        new_key = key.replace('matcher.joint_mamba.', 'matcher.hla_interaction.')
        if new_key.startswith('module.'):
            new_key = new_key[len('module.'):]
        if new_key in normalized:
            raise ValueError(f'Duplicate key after checkpoint conversion: {new_key}')
        normalized[new_key] = value
    return normalized
