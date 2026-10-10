"""Opt-in, short-run training diagnostics for PFM ablations.

Enable with HLA_PFM_DIAG=1. This helper records only Python scalars, never
keeps graph tensors between batches. It is intended for short debug runs:
extracting GPU scalars at every step introduces synchronization overhead.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch


class PFMTrainingDiagnostics:
    def __init__(self):
        self.enabled = os.environ.get("HLA_PFM_DIAG", "0").lower() in {"1", "true", "yes"}
        self.every = max(1, int(os.environ.get("HLA_PFM_LOG_EVERY", "1")))
        self.output_dir = Path(os.environ.get("HLA_PFM_DIAG_DIR", "pfm_diagnostics"))
        self.active = False
        self.record = None

    def start_step(self, step: int, batch_idx: int, mode: str, amp_enabled: bool):
        self.active = self.enabled and step % self.every == 0
        self.record = None
        if self.active:
            self.record = {
                "global_step": int(step), "batch_idx": int(batch_idx),
                "pfm_mode": mode, "amp_enabled": bool(amp_enabled),
                "attention_calls": [], "first_nonfinite": None,
            }

    def begin_attention(self, label: str):
        if not self.active:
            return None
        entry = {"label": str(label), "call_index": len(self.record["attention_calls"])}
        self.record["attention_calls"].append(entry)
        return entry

    def tensor(self, entry, name: str, value: torch.Tensor, near_zero: float | None = None):
        if entry is None:
            return
        with torch.no_grad():
            x = value.detach()
            if x.is_complex():
                x = torch.view_as_real(x)
            x = x.float()
            finite = torch.isfinite(x)
            nonfinite_count = int((~finite).sum().item())
            stats = {"nonfinite_count": nonfinite_count}
            if x.numel():
                stats["negative_fraction"] = float(((x < 0) & finite).float().mean().item())
                valid = x[finite]
                if valid.numel():
                    stats["min"] = float(valid.min().item())
                    stats["max"] = float(valid.max().item())
                if near_zero is not None:
                    stats["near_zero_fraction"] = float(((x.abs() <= near_zero) & finite).float().mean().item())
            entry[name] = stats
            if nonfinite_count and self.record["first_nonfinite"] is None:
                self.record["first_nonfinite"] = f"{entry['label']}.{name}"

    def denominator(self, entry, den_raw: torch.Tensor, eps: float):
        if entry is None:
            return
        self.tensor(entry, "den_raw", den_raw)
        with torch.no_grad():
            x = den_raw.detach().float()
            finite = x[torch.isfinite(x)]
            if finite.numel():
                entry["den_raw"]["p01"] = float(torch.quantile(finite, 0.01).item())
                entry["den_raw"]["clamped_fraction"] = float((finite <= eps).float().mean().item())
                entry["den_raw"]["negative_fraction"] = float((finite < 0).float().mean().item())

    def _batch_scalars(self, batch):
        if self.record is None:
            return
        for name, key in (("coarse_predicted_matches", "mconf"),
                          ("fine_predicted_matches", "mconf_f"),
                          ("fine_training_samples_including_gt_padding", "b_ids")):
            value = batch.get(key)
            if torch.is_tensor(value):
                self.record[name] = int(value.numel())
        losses = batch.get("loss_scalars", {})
        def safe_loss(value):
            result = float(value.item()) if torch.is_tensor(value) else float(value)
            return result if math.isfinite(result) else str(result)
        self.record["losses"] = {
            str(name): safe_loss(value)
            for name, value in losses.items()
        }

    def forward_complete(self, batch):
        if not self.active:
            return
        self._batch_scalars(batch)
        loss = batch.get("loss")
        if torch.is_tensor(loss) and not bool(torch.isfinite(loss.detach()).all().item()):
            self.record["status"] = "nonfinite_loss"
            if self.record["first_nonfinite"] is None:
                self.record["first_nonfinite"] = "loss"
            self._write()
            raise FloatingPointError("PFM diagnostic: nonfinite training loss; see the JSONL log")

    def after_backward(self, model):
        if not self.active:
            return
        total_sq = 0.0
        hla_sq = 0.0
        nonfinite_count = 0
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().float()
                finite = torch.isfinite(grad)
                nonfinite_count += int((~finite).sum().item())
                valid = grad[finite]
                if valid.numel():
                    squared = float(valid.double().square().sum().item())
                    total_sq += squared
                    if name.startswith("matcher.hla_interaction."):
                        hla_sq += squared
        self.record["gradient_global_l2"] = math.sqrt(total_sq)
        self.record["gradient_hla_interaction_l2"] = math.sqrt(hla_sq)
        self.record["gradient_nonfinite_count"] = nonfinite_count
        self.record["gradient_may_be_amp_scaled"] = self.record["amp_enabled"]
        self.record["status"] = "nonfinite_gradient" if nonfinite_count else "ok"
        self._write()
        if nonfinite_count:
            raise FloatingPointError("PFM diagnostic: nonfinite gradient; see the JSONL log")

    def exception(self, batch, exc: Exception):
        if not self.active:
            return
        self._batch_scalars(batch)
        self.record["status"] = "exception"
        self.record["exception_type"] = type(exc).__name__
        self.record["exception_message"] = str(exc)[:500]
        self._write()

    def _write(self):
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / f"pfm_training_rank{rank}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.record, allow_nan=False) + "\n")
        self.active = False
        self.record = None
