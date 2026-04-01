import os
import time
from pathlib import Path

import torch

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as stats_utils

trainer_stats_name = "noop"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device")
    if device is None:
        device = torch.get_default_device()
    out = kwargs.get("baseline_output_path")
    if out is None:
        name = stats_utils.apply_trainer_stats_file_prefix_to_basename(
            stats_utils.train_duration_basename(), conf
        )
        out_dir = (
            os.environ.get("BASELINE_TRAIN_OUTPUT_DIR", "").strip()
            or os.environ.get("WHISPER_OUTPUT_DIR", "").strip()
        )
        out = Path(out_dir) / name if out_dir else Path(name)
    else:
        out = Path(out)
    return NOOPTrainerStats(device=device, output_path=out)


class NOOPTrainerStats(base.TrainerStats):
    """Baseline wall time for the full training loop (``start_train`` … ``stop_train``).

    Uses ``torch.cuda.synchronize(device)`` before each timestamp on CUDA so the
    interval aligns with GPU-visible start/end of training. ``log_stats`` writes
    one line, ``duration_ms <float>``, to ``output_path``. Default basename:
    ``baseline_train_duration.txt`` (``memory_``-prefixed when synthetic_whisper effective ``data_type`` is ``memory``),
    or ``baseline_train_duration_run_<index>.txt`` when ``RUN_REPEAT_INDEX`` is set.
    If ``WHISPER_OUTPUT_DIR`` or ``BASELINE_TRAIN_OUTPUT_DIR`` is set (e.g. exported
    from ``scripts/start-whisper.sh``), files go under that directory (typically
    ``results/data/batch_<batch>_worker_<workers>``). Override fully with
    ``baseline_output_path`` in ``construct_trainer_stats`` kwargs.
    """

    def __init__(self, device: torch.device, output_path: Path) -> None:
        super().__init__()
        self.device = device
        self._output_path = output_path
        self._train_start_ns: int | None = None
        self._train_duration_ns: int | None = None

    def start_train(self) -> None:
        self._train_duration_ns = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._train_start_ns = time.perf_counter_ns()

    def stop_train(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self._train_start_ns is not None:
            self._train_duration_ns = time.perf_counter_ns() - self._train_start_ns
        self._train_start_ns = None

    def start_step(self) -> None:
        pass

    def stop_step(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_forward(self) -> None:
        pass

    def stop_forward(self) -> None:
        pass

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        d = self._train_duration_ns
        if d is None:
            return
        path = self._output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        ms = d / 1e6
        with path.open("w", encoding="utf-8") as f:
            f.write(f"duration_ms {ms}\n")

    def log_loss(self, loss: torch.Tensor) -> None:
        pass
