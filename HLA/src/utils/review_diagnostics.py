"""Opt-in timing and FLOP diagnostics for the HLA/HLA test path.

The module is inert unless HLA_REVIEW_TIMING=1 or HLA_REVIEW_FLOPS=1.
CUDA event timing is collected asynchronously and synchronized once per batch.
FLOP accounting counts Conv/Linear module MACs and explicitly counts functional
attention products plus the complex/PFM scalar arithmetic on the real test pass.
"""

from __future__ import annotations

import contextvars
import json
import os
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
from torch import nn


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except (TypeError, ValueError):
        return 0


class ReviewDiagnostics:
    """Collect CUDA event timings and write per-batch JSONL records."""

    def __init__(self, output_dir: Optional[str] = None):
        self.timing_enabled = _env_flag("HLA_REVIEW_TIMING")
        self.flops_enabled = _env_flag("HLA_REVIEW_FLOPS")
        if self.timing_enabled and self.flops_enabled:
            raise ValueError(
                "Run timing and FLOPs as separate passes; FLOP hooks add Python overhead to inference."
            )
        self.output_dir = Path(
            output_dir or os.environ.get("HLA_REVIEW_DIR", "review_diagnostics")
        )
        if self.timing_enabled and not torch.cuda.is_available():
            raise RuntimeError("HLA_REVIEW_TIMING=1 requires a CUDA-enabled PyTorch runtime.")
        self._events = []
        self._batch_idx = None
        self._pair_count = 0
        self._batch_metadata = {}
        self._stage_per_pair = defaultdict(list)
        self._total_pairs = 0
        self._expected_input_hw = None

    def start_batch(self, batch_idx: int, pair_count: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._events = []
        self._batch_idx = int(batch_idx)
        self._pair_count = max(int(pair_count), 1)
        self._batch_metadata = dict(metadata or {})
        if self.timing_enabled or self.flops_enabled:
            height = self._batch_metadata.get("input_height")
            width = self._batch_metadata.get("input_width")
            if height is not None and width is not None:
                current_hw = (int(height), int(width))
                if self._expected_input_hw is None:
                    self._expected_input_hw = current_hw
                elif current_hw != self._expected_input_hw:
                    raise ValueError(
                        "Review diagnostics expect the fixed test size {}; got {} at batch {}.".format(
                            self._expected_input_hw, current_hw, batch_idx
                        )
                    )

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.timing_enabled:
            with nullcontext():
                yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events.append((str(name), start, end))

    def finish_batch(self, synchronized: bool = False) -> Dict[str, float]:
        if not self.timing_enabled:
            return {}
        if not synchronized:
            torch.cuda.synchronize()
        stage_ms = defaultdict(float)
        for name, start, end in self._events:
            stage_ms[name] += float(start.elapsed_time(end))
        per_pair = {name: value / self._pair_count for name, value in stage_ms.items()}
        if "matcher_total" in stage_ms:
            accounted = sum(stage_ms.get(name, 0.0) for name in (
                "hla_interaction", "coarse_matching", "fine_matching"
            ))
            stage_ms["matcher_other"] = stage_ms["matcher_total"] - accounted
            per_pair["matcher_other"] = stage_ms["matcher_other"] / self._pair_count

        for name, value in per_pair.items():
            self._stage_per_pair[name].append(value)
        self._total_pairs += self._pair_count
        record = {
            "batch_idx": self._batch_idx,
            "pair_count": self._pair_count,
            "metadata": self._batch_metadata,
            "stage_ms_per_batch": dict(stage_ms),
            "stage_ms_per_pair": per_pair,
            "note": "Nested HLA timings are included in hla_interaction; do not add parent and child stages.",
        }
        self._append_jsonl(self.output_dir / f"timing_rank{_rank()}.jsonl", record)
        return dict(stage_ms)

    def summary(self) -> Dict[str, Any]:
        stages = {}
        for name, values in self._stage_per_pair.items():
            tensor = torch.tensor(values, dtype=torch.float64)
            stages[name] = {
                "mean_ms_per_pair": float(tensor.mean().item()),
                "median_ms_per_pair": float(tensor.median().item()),
                "p90_ms_per_pair": float(torch.quantile(tensor, 0.90).item()),
            }
        return {"pair_count": self._total_pairs, "stages": stages}

    @staticmethod
    def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")


class ReviewFlopCounter:
    """Count module MACs and functional FLOPs during one real test forward."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self._handles = []
        self._module_macs = defaultdict(float)
        self._functional_macs = defaultdict(float)
        self._scalar_flops = defaultdict(float)
        self._batch_idx = None
        self._pair_count = 1
        self._stage_flops_sum = defaultdict(float)
        self._total_pairs = 0
        self._total_flops = 0.0

    def attach(self, module: nn.Module, prefix: str) -> None:
        """Register hooks on arithmetic modules; shared modules count each call."""
        mac_types = (
            nn.Linear,
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        )
        scalar_types = (
            nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm,
            nn.ReLU, nn.GELU, nn.SiLU, nn.PReLU, nn.Softmax, nn.Upsample
        )
        for module_name, child in module.named_modules():
            full_name = prefix if not module_name else f"{prefix}.{module_name}"
            custom_convnext_layer = (
                child.__class__.__module__.startswith("src.convnextv2")
                and child.__class__.__name__ in {"LayerNorm", "GRN"}
            )
            custom_matcher_layer = (
                child.__class__.__module__.startswith("src.hla.utils")
                and child.__class__.__name__ in {"up_conv4", "MLPMixerEncoderLayer"}
            )
            convnext_block = (
                full_name.startswith("backbone.cnn.stages")
                and child.__class__.__name__ == "Block"
            )
            if (not isinstance(child, mac_types + scalar_types) and not custom_convnext_layer
                    and not custom_matcher_layer and not convnext_block):
                continue
            self._handles.append(
                child.register_forward_hook(self._module_hook(full_name))
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _module_hook(self, name: str):
        def hook(module, inputs, output):
            if _ACTIVE_FLOP_COUNTER.get() is not self or not torch.is_tensor(output):
                return
            if isinstance(module, nn.Linear):
                macs = output.numel() * module.in_features
                self._module_macs[name] += float(macs)
                if module.bias is not None:
                    self._scalar_flops[f"{name}.bias_add"] += float(output.numel())
            elif isinstance(module, (
                nn.Conv1d, nn.Conv2d, nn.Conv3d,
                nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            )):
                kernel_size = 1
                for k in module.kernel_size:
                    kernel_size *= k
                macs = output.numel() * (module.in_channels // module.groups) * kernel_size
                self._module_macs[name] += float(macs)
                if module.bias is not None:
                    self._scalar_flops[f"{name}.bias_add"] += float(output.numel())
            elif isinstance(module, nn.LayerNorm):
                self._scalar_flops[f"{name}.layer_norm"] += 8.0 * output.numel()
            elif isinstance(module, nn.GroupNorm):
                self._scalar_flops[f"{name}.group_norm"] += 8.0 * output.numel()
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                # Approximate inference BatchNorm as centering, scaling, and affine.
                self._scalar_flops[f"{name}.batch_norm"] += 4.0 * output.numel()
            elif isinstance(module, nn.GELU):
                self._scalar_flops[f"{name}.gelu"] += 8.0 * output.numel()
            elif isinstance(module, nn.SiLU):
                self._scalar_flops[f"{name}.silu"] += 4.0 * output.numel()
            elif isinstance(module, (nn.ReLU, nn.PReLU)):
                self._scalar_flops[f"{name}.relu"] += float(output.numel())
            elif isinstance(module, nn.Softmax):
                self._scalar_flops[f"{name}.softmax"] += 5.0 * output.numel()
            elif isinstance(module, nn.Upsample):
                if module.mode == "bilinear":
                    self._scalar_flops[f"{name}.bilinear_interpolation"] += 7.0 * output.numel()
            elif module.__class__.__name__ == "LayerNorm":
                self._scalar_flops[f"{name}.layer_norm"] += 8.0 * output.numel()
            elif module.__class__.__name__ == "GRN":
                self._scalar_flops[f"{name}.grn"] += 8.0 * output.numel()
            elif module.__class__.__name__ == "MLPMixerEncoderLayer":
                self._scalar_flops[f"{name}.residual_adds"] += 2.0 * output.numel()
            elif module.__class__.__name__ == "up_conv4":
                self._scalar_flops[f"{name}.branch_add"] += float(output.numel())
            elif module.__class__.__name__ == "Block" and name.startswith("backbone.cnn.stages"):
                self._scalar_flops[f"{name}.residual_add"] += float(output.numel())

        return hook

    @staticmethod
    def _stage_for_name(name: str) -> str:
        if name.startswith("backbone"):
            return "backbone"
        if name.startswith("hla_interaction"):
            return "hla_interaction"
        if name.startswith("coarse_matching"):
            return "coarse_matching"
        if name.startswith(("fine_matching", "fine_", "up2", "up3", "conv7", "conv8", "act")):
            return "fine_matching"
        if name.startswith(("hla.", "mlla.")):
            return "hla_interaction"
        return "matcher_other"

    def start_batch(self, batch_idx: int, pair_count: int) -> None:
        self._module_macs.clear()
        self._functional_macs.clear()
        self._scalar_flops.clear()
        self._batch_idx = int(batch_idx)
        self._pair_count = max(int(pair_count), 1)

    def add_macs(self, name: str, count: int | float) -> None:
        self._functional_macs[name] += float(count)

    def add_scalar_flops(self, name: str, count: int | float) -> None:
        self._scalar_flops[name] += float(count)

    def finish_batch(
        self, input_height: int, input_width: int, parameter_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        module_by_stage = defaultdict(float)
        functional_by_stage = defaultdict(float)
        scalar_by_stage = defaultdict(float)
        for name, value in self._module_macs.items():
            stage = self._stage_for_name(name.replace("matcher.", "", 1))
            module_by_stage[stage] += value
        for name, value in self._functional_macs.items():
            functional_by_stage[self._stage_for_name(name)] += value
        for name, value in self._scalar_flops.items():
            scalar_by_stage[self._stage_for_name(name)] += value

        stages = {}
        for stage in set(module_by_stage) | set(functional_by_stage) | set(scalar_by_stage):
            stages[stage] = {
                "module_macs": module_by_stage[stage],
                "functional_macs": functional_by_stage[stage],
                "scalar_flops": scalar_by_stage[stage],
                "estimated_flops": (
                    2.0 * (module_by_stage[stage] + functional_by_stage[stage])
                    + scalar_by_stage[stage]
                ),
            }
            self._stage_flops_sum[stage] += stages[stage]["estimated_flops"]

        total_module_macs = sum(self._module_macs.values())
        total_functional_macs = sum(self._functional_macs.values())
        total_scalar_flops = sum(self._scalar_flops.values())
        total_flops = 2.0 * (total_module_macs + total_functional_macs) + total_scalar_flops
        record = {
            "batch_idx": self._batch_idx,
            "pair_count": self._pair_count,
            "input_height": int(input_height),
            "input_width": int(input_width),
            "metadata": dict(metadata or {}),
            "module_macs": total_module_macs,
            "functional_attention_macs": total_functional_macs,
            "scalar_flops": total_scalar_flops,
            "estimated_total_flops": total_flops,
            "estimated_gflops_per_pair": total_flops / self._pair_count / 1e9,
            "parameter_count": parameter_count,
            "stages": stages,
            "counting_convention": (
                "1 MAC = 2 FLOPs. Conv/Linear MACs use actual output shapes; "
                "attention matmuls/einsums and PFM complex arithmetic are counted explicitly. "
                "Complex integer power uses repeated complex multiply (6 scalar FLOPs each); "
                "Softplus is estimated as exp+add+log; LayerNorm/GroupNorm/GRN=8, "
                "inference BatchNorm=4, GELU=8, SiLU=4, "
                "Softmax=5, bilinear interpolation=7, and ReLU=1 scalar operations per output. "
                "Reshape, concatenate, indexing, comparisons, and memory movement are excluded."
            ),
        }
        self._total_pairs += self._pair_count
        self._total_flops += total_flops
        for stage, values in stages.items():
            values["estimated_gflops_per_pair"] = values["estimated_flops"] / self._pair_count / 1e9
        ReviewDiagnostics._append_jsonl(
            self.output_dir / f"flops_rank{_rank()}.jsonl", record
        )
        return record

    def summary(self) -> Dict[str, Any]:
        stages = {
            name: value / max(self._total_pairs, 1)
            for name, value in self._stage_flops_sum.items()
        }
        return {
            "pair_count": self._total_pairs,
            "mean_gflops_per_pair": self._total_flops / max(self._total_pairs, 1) / 1e9,
            "mean_gflops_per_pair_by_stage": {
                name: value / 1e9 for name, value in stages.items()
            },
        }


_ACTIVE_FLOP_COUNTER = contextvars.ContextVar("hla_review_flop_counter", default=None)


@contextmanager
def collect_review_flops(counter: ReviewFlopCounter) -> Iterator[None]:
    token = _ACTIVE_FLOP_COUNTER.set(counter)
    try:
        yield
    finally:
        _ACTIVE_FLOP_COUNTER.reset(token)


def add_review_macs(name: str, count: int | float) -> None:
    counter = _ACTIVE_FLOP_COUNTER.get()
    if counter is not None:
        counter.add_macs(name, count)


def add_review_flops(name: str, count: int | float) -> None:
    counter = _ACTIVE_FLOP_COUNTER.get()
    if counter is not None:
        counter.add_scalar_flops(name, count)
