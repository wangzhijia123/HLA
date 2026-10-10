"""Lightweight profiling adapters used by the HLA model and scripts.

The default pass-through profiler deliberately has no PyTorch Lightning import,
so the standalone harmonic attention module can be tested on CPU.
"""
from contextlib import contextmanager
import torch


class PassThroughProfiler:
    @contextmanager
    def profile(self, action_name):
        yield action_name

    def summary(self):
        return ''


class InferenceProfiler:
    def __init__(self):
        try:
            from pytorch_lightning.profiler import SimpleProfiler
        except ImportError:
            from pytorch_lightning.profilers import SimpleProfiler
        self.profiler = SimpleProfiler()

    @contextmanager
    def profile(self, action_name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.profiler.start(action_name)
        try:
            yield action_name
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.profiler.stop(action_name)

    def summary(self):
        return self.profiler.summary()


def build_profiler(name):
    if name is None:
        return PassThroughProfiler()
    if name == 'inference':
        return InferenceProfiler()
    if name == 'pytorch':
        try:
            from pytorch_lightning.profiler import PyTorchProfiler
        except ImportError:
            from pytorch_lightning.profilers import PyTorchProfiler
        return PyTorchProfiler(use_cuda=torch.cuda.is_available(), profile_memory=True, row_limit=100)
    raise ValueError(f'Invalid profiler: {name}')
