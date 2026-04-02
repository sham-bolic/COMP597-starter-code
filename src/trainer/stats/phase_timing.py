"""CUDA-synchronized wall-time stats for training steps and sub-phases.

``trainer_stats_configs.phase_timing.measure_phase`` selects what is timed per run:

- ``all`` — whole step plus forward, backward, optimizer (default).
- ``step``, ``forward``, ``backward``, ``optimizer`` — only that interval gets
  ``torch.cuda.synchronize`` + ``perf_counter``; other hooks are no-ops so you
  measure one phase at a time without extra sync boundaries.

Writes two CSVs per run under ``output_dir``; all durations are in **milliseconds**
(column names ``*_ms``). Each step also records **resource** columns (after CUDA
sync when on GPU): ``gpu_util`` (NVML %), ``cuda_mem_allocated_gb`` and
``cuda_mem_reserved_gb`` (``torch.cuda``; decimal GB), and ``cpu_util`` (% for this process, same
as ``resource_util``). When
``measure_phase`` is not ``all``, filenames include ``phase_timing_measure_<phase>_``
so separate runs do not overwrite each other.

Forward timing includes anything inside ``Trainer.forward`` (e.g.
``optimizer.zero_grad()`` for ``SimpleTrainer``).
"""

from __future__ import annotations

import csv
import logging
import math
import os
import time
from typing import Any

import numpy as np
import pynvml
import psutil
import torch

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as stats_utils
from src.trainer.stats.utils import RunningTimer

logger = logging.getLogger(__name__)

trainer_stats_name = "phase_timing"

_MEASURE_PHASE_CHOICES = frozenset({"all", "step", "forward", "backward", "optimizer"})

_SUMMARY_TIMING_COLS = ("step_ms", "forward_ms", "backward_ms", "optimizer_ms")
_SUMMARY_RESOURCE_COLS = (
    "gpu_util",
    "cuda_mem_allocated_gb",
    "cuda_mem_reserved_gb",
    "cpu_util",
)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning(
            "No device provided to phase_timing trainer stats. Using default PyTorch device"
        )
        device = torch.get_default_device()
    fp = stats_utils.trainer_stats_file_prefix(conf)
    cfg = conf.trainer_stats_configs.phase_timing
    measure = str(getattr(cfg, "measure_phase", "all")).strip().lower()
    if measure not in _MEASURE_PHASE_CHOICES:
        raise ValueError(
            f"phase_timing.measure_phase must be one of {sorted(_MEASURE_PHASE_CHOICES)}, got {measure!r}"
        )
    if device.type == "cuda" and torch.cuda.is_available() and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return PhaseTimingStats(
        device=device,
        run_num=cfg.run_num,
        output_dir=cfg.output_dir,
        warmup_steps=max(0, int(cfg.warmup_steps)),
        measure_phase=measure,
        file_prefix=fp,
    )


class PhaseTimingStats(base.TrainerStats):
    def __init__(
        self,
        device: torch.device,
        run_num: int,
        output_dir: str,
        warmup_steps: int,
        measure_phase: str = "all",
        file_prefix: str = "",
    ) -> None:
        super().__init__()
        self.device = device
        self.run_num = run_num
        self.output_dir = output_dir
        self.warmup_steps = warmup_steps
        self._measure_phase = measure_phase
        self._file_prefix = file_prefix

        self.step_stats = RunningTimer()
        self.forward_stats = RunningTimer()
        self.backward_stats = RunningTimer()
        self.optimizer_step_stats = RunningTimer()

        self._per_step_rows: list[dict[str, Any]] = []
        self._wrote_files = False
        self._train_start_ns: int | None = None
        self._train_duration_ns: int | None = None

        self._process = psutil.Process()
        self._gpu_handle = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                gpu_index = (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
            except Exception as e:
                logger.warning(
                    "phase_timing: NVML not available (%s); gpu_util will be NaN",
                    e,
                )

        gid = self.device.index
        rank_label = "cpu" if gid is None else str(gid)
        self._run_token = f"{file_prefix}run_{run_num}_"
        if self._measure_phase == "all":
            mid = "phase_timing"
        else:
            mid = f"phase_timing_measure_{self._measure_phase}"
        self._base_name = f"{self._run_token}{mid}_rank_{rank_label}"
        os.makedirs(self.output_dir, exist_ok=True)
        self._per_step_path = os.path.join(
            self.output_dir, f"{self._base_name}_per_step.csv"
        )
        self._summary_path = os.path.join(
            self.output_dir, f"{self._base_name}_summary.csv"
        )

    def _want(self, part: str) -> bool:
        return self._measure_phase == "all" or self._measure_phase == part

    @staticmethod
    def _ns_to_ms_or_nan(stats: RunningTimer) -> float:
        """Last interval in milliseconds, or NaN if no measurement was recorded."""
        if stats.get_last() < 0:
            return float("nan")
        return stats.get_last() / 1.0e6

    def _sample_resources(self) -> dict[str, float]:
        """GPU util (NVML) / CUDA allocator memory / CPU util at end of step."""
        gpu_u = float("nan")
        cuda_alloc_gb = float("nan")
        cuda_reserved_gb = float("nan")
        if self.device.type == "cuda" and torch.cuda.is_available():
            _sync_device(self.device)
            try:
                cuda_alloc_gb = torch.cuda.memory_allocated(self.device) / 1e9
                cuda_reserved_gb = torch.cuda.memory_reserved(self.device) / 1e9
            except Exception as e:
                logger.debug("phase_timing: CUDA memory query failed: %s", e)
            if self._gpu_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    gpu_u = float(util.gpu)
                except Exception as e:
                    logger.debug("phase_timing: NVML sample failed: %s", e)
        cpu_u = float(self._process.cpu_percent())
        return {
            "gpu_util": gpu_u,
            "cuda_mem_allocated_gb": cuda_alloc_gb,
            "cuda_mem_reserved_gb": cuda_reserved_gb,
            "cpu_util": cpu_u,
        }

    def start_train(self) -> None:
        self._train_duration_ns = None
        _sync_device(self.device)
        self._train_start_ns = time.perf_counter_ns()

    def stop_train(self) -> None:
        _sync_device(self.device)
        if self._train_start_ns is not None:
            self._train_duration_ns = time.perf_counter_ns() - self._train_start_ns
        self._train_start_ns = None

    def start_step(self) -> None:
        if not self._want("step"):
            return
        _sync_device(self.device)
        self.step_stats.start()

    def stop_step(self) -> None:
        if self._want("step"):
            _sync_device(self.device)
            self.step_stats.stop()
        row = {
            "step_index": len(self._per_step_rows),
            "step_ms": self._ns_to_ms_or_nan(self.step_stats),
            "forward_ms": self._ns_to_ms_or_nan(self.forward_stats),
            "backward_ms": self._ns_to_ms_or_nan(self.backward_stats),
            "optimizer_ms": self._ns_to_ms_or_nan(self.optimizer_step_stats),
        }
        row.update(self._sample_resources())
        self._per_step_rows.append(row)

    def start_forward(self) -> None:
        if not self._want("forward"):
            return
        _sync_device(self.device)
        self.forward_stats.start()

    def stop_forward(self) -> None:
        if not self._want("forward"):
            return
        _sync_device(self.device)
        self.forward_stats.stop()

    def start_backward(self) -> None:
        if not self._want("backward"):
            return
        _sync_device(self.device)
        self.backward_stats.start()

    def stop_backward(self) -> None:
        if not self._want("backward"):
            return
        _sync_device(self.device)
        self.backward_stats.stop()

    def start_optimizer_step(self) -> None:
        if not self._want("optimizer"):
            return
        _sync_device(self.device)
        self.optimizer_step_stats.start()

    def stop_optimizer_step(self) -> None:
        if not self._want("optimizer"):
            return
        _sync_device(self.device)
        self.optimizer_step_stats.stop()

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def log_step(self) -> None:
        pass

    def _build_summary(self) -> dict[str, Any]:
        cols = _SUMMARY_TIMING_COLS + _SUMMARY_RESOURCE_COLS
        n = len(self._per_step_rows)
        warm = min(self.warmup_steps, n)
        body = self._per_step_rows[warm:]
        m = max(0, n - warm)
        total_train_ms = (
            float(self._train_duration_ns) / 1.0e6
            if self._train_duration_ns is not None
            else float("nan")
        )

        def _finite_sum(key: str) -> float:
            vals = [float(r[key]) for r in self._per_step_rows if math.isfinite(float(r[key]))]
            return float(sum(vals)) if vals else float("nan")

        sum_step_ms = _finite_sum("step_ms") if n else float("nan")
        out: dict[str, Any] = {
            "measure_phase": self._measure_phase,
            "warmup_steps": self.warmup_steps,
            "n_steps_total": n,
            "n_steps_summarized": m,
            "total_train_ms": total_train_ms,
            "sum_step_ms": sum_step_ms,
        }
        if not body:
            for c in cols:
                out[f"mean_{c}"] = float("nan")
                out[f"std_{c}"] = float("nan")
            return out
        arr = np.array(
            [[float(r[c]) if math.isfinite(float(r[c])) else np.nan for c in cols] for r in body],
            dtype=np.float64,
        )
        for j, c in enumerate(cols):
            col = arr[:, j]
            fin = col[np.isfinite(col)]
            if fin.size == 0:
                out[f"mean_{c}"] = float("nan")
                out[f"std_{c}"] = float("nan")
            elif fin.size == 1:
                out[f"mean_{c}"] = float(fin[0])
                out[f"std_{c}"] = float("nan")
            else:
                out[f"mean_{c}"] = float(np.mean(fin))
                out[f"std_{c}"] = float(np.std(fin, ddof=0))
        return out

    def _write_csvs(self) -> None:
        if self._wrote_files:
            return
        fieldnames = [
            "step_index",
            "step_ms",
            "forward_ms",
            "backward_ms",
            "optimizer_ms",
            "gpu_util",
            "cuda_mem_allocated_gb",
            "cuda_mem_reserved_gb",
            "cpu_util",
        ]
        with open(self._per_step_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self._per_step_rows:
                w.writerow(
                    {
                        k: ("" if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in r.items()
                    }
                )

        summary = self._build_summary()
        sum_fields = list(summary.keys())
        with open(self._summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sum_fields)
            w.writeheader()
            w.writerow(summary)

        self._wrote_files = True
        logger.info("Wrote phase timing %s and %s", self._per_step_path, self._summary_path)

    def log_stats(self) -> None:
        self._write_csvs()
        s = self._build_summary()
        parts = [
            f"measure_phase={s.get('measure_phase')!r}",
            f"total_train_ms={s.get('total_train_ms', float('nan')):.6g}",
            f"sum_step_ms={s.get('sum_step_ms', float('nan')):.6g}",
        ]
        for key, label in (
            ("forward", "forward"),
            ("backward", "backward"),
            ("optimizer", "optimizer"),
        ):
            if self._measure_phase in ("all", key):
                mean_v = s.get(f"mean_{key}_ms", float("nan"))
                std_v = s.get(f"std_{key}_ms", float("nan"))
                parts.append(f"{label} mean_ms={mean_v:.6g} std_ms={std_v:.6g}")
        if self._measure_phase in ("all", "step"):
            parts.append(
                f"step mean_ms={s.get('mean_step_ms', float('nan')):.6g} "
                f"std_ms={s.get('std_step_ms', float('nan')):.6g}"
            )
        parts.append(
            f"gpu_util mean={s.get('mean_gpu_util', float('nan')):.4g}% "
            f"cuda_mem_alloc mean={s.get('mean_cuda_mem_allocated_gb', float('nan')):.4g} GB "
            f"cuda_mem_resv mean={s.get('mean_cuda_mem_reserved_gb', float('nan')):.4g} GB "
            f"cpu_util mean={s.get('mean_cpu_util', float('nan')):.4g}%"
        )
        parts.append(f"(warmup_steps={s.get('warmup_steps')}, n_steps={s.get('n_steps_summarized')})")
        print("Phase timing (summarized, ms): " + " | ".join(parts))

        n = len(self._per_step_rows)
        warm = min(self.warmup_steps, n)
        body = self._per_step_rows[warm:]
        cols = _SUMMARY_TIMING_COLS + _SUMMARY_RESOURCE_COLS
        cov: list[str] = []
        if body:
            total = len(body)
            for c in cols:
                nf = sum(1 for r in body if math.isfinite(float(r[c])))
                if c in _SUMMARY_TIMING_COLS:
                    # All-NaN is normal for unmeasured phases when measure_phase is not "all".
                    incomplete = 0 < nf < total
                elif c in ("gpu_util", "cuda_mem_allocated_gb", "cuda_mem_reserved_gb"):
                    incomplete = (
                        self.device.type == "cuda"
                        and torch.cuda.is_available()
                        and nf < total
                    )
                else:
                    # cpu_util: expect a value every step; any mix of NaN is suspicious.
                    incomplete = 0 < nf < total
                if incomplete:
                    cov.append(f"{c}: {nf}/{total} finite steps (excluded NaN from mean/std)")
        if cov:
            print("phase_timing: incomplete metrics — " + "; ".join(cov))
