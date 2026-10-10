from src.hla.checkpoints import load_hla_state_dict
import torch


def test_local_checkpoint_key_migration(tmp_path):
    path = tmp_path / 'model.ckpt'
    torch.save({'state_dict': {'matcher.joint_mamba.atten.weight': torch.ones(2),
                               'backbone.cnn.weight': torch.zeros(1)}}, path)
    sd = load_hla_state_dict(path)
    assert 'matcher.hla_interaction.atten.weight' in sd
    assert 'backbone.cnn.weight' in sd
