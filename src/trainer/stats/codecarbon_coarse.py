"""CodeCarbon with coarse step tasks only (no forward/backward/optimizer substeps).

The full-run CSV is named ``cc_full_coarse_…`` (distinct from ``cc_full_…``).
The step-task CSV is ``cc_step_coarse_…`` with ``-coarse_steps.csv`` task rows
(one CodeCarbon task every ``step_interval`` training steps; default 10).
There is no substep tracker.

``log_stats`` writes ``cc_coarse_run_summary_…csv`` with CUDA-sync wall
``total_train_wall_s`` and CodeCarbon chunk ``duration`` summed / averaged back
to per-step seconds.
"""

from __future__ import annotations

import logging
import os
import time

import codecarbon
import codecarbon.core.cpu
import numpy as np
import pandas as pd
import torch
from codecarbon import OfflineEmissionsTracker

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as stats_utils
from src.trainer.stats.codecarbon import SimpleFileOutput

logger = logging.getLogger(__name__)

codecarbon.core.cpu.is_psutil_available = lambda: False

trainer_stats_name = "codecarbon_coarse"


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning(
            "No device provided to codecarbon_coarse trainer stats. Using default PyTorch device"
        )
        device = torch.get_default_device()
    fp = stats_utils.trainer_stats_file_prefix(conf)
    cfg = conf.trainer_stats_configs.codecarbon_coarse
    step_interval = int(cfg.step_interval)
    if step_interval < 1:
        raise ValueError(f"codecarbon_coarse.step_interval must be >= 1, got {step_interval}")
    return CodeCarbonCoarseStats(
        device,
        cfg.run_num,
        cfg.project_name,
        cfg.output_dir,
        step_interval=step_interval,
        file_prefix=fp,
    )


class CodeCarbonCoarseStats(base.TrainerStats):
    def __init__(
        self,
        device: torch.device,
        run_num: int,
        project_name: str,
        output_dir: str,
        step_interval: int,
        file_prefix: str = "",
    ) -> None:
        self.iteration = 0
        self.device = device
        self.run_num = run_num
        self._file_prefix = file_prefix
        self.step_interval = step_interval
        self.project_name = project_name
        self.output_dir = output_dir
        self.losses: list = []

        self._chunk_task_name: str | None = None
        self._chunk_open = False
        self._train_start_ns: int | None = None
        self._train_duration_ns: int | None = None
        self._partial_final_chunk = False

        gpu_id = self.device.index
        run_number = f"{file_prefix}run_{run_num}_"
        self._run_number = run_number
        self._gpu_id = gpu_id
        os.makedirs(self.output_dir, exist_ok=True)

        self.total_training_tracker = OfflineEmissionsTracker(
            project_name=project_name,
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[
                SimpleFileOutput(
                    output_file_name=f"{run_number}cc_full_coarse_rank_{gpu_id}.csv",
                    output_dir=output_dir,
                )
            ],
            allow_multiple_runs=True,
            log_level="warning",
            gpu_ids=[gpu_id],
        )

        self.training_step_tracker = OfflineEmissionsTracker(
            project_name=project_name,
            experiment_name="coarse_steps",
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[
                SimpleFileOutput(
                    output_file_name=f"{run_number}cc_step_coarse_rank_{gpu_id}.csv",
                    output_dir=output_dir,
                )
            ],
            allow_multiple_runs=True,
            api_call_interval=-1,
            gpu_ids=[gpu_id],
            log_level="warning",
        )

        self.training_step_tracker.start()

    def start_train(self) -> None:
        self._train_duration_ns = None
        self._partial_final_chunk = False
        _sync_device(self.device)
        self._train_start_ns = time.perf_counter_ns()
        self.total_training_tracker.start()

    def stop_train(self) -> None:
        _sync_device(self.device)
        if self.iteration % self.step_interval != 0 and self.iteration > 0:
            self._partial_final_chunk = True
        if self._chunk_open and self._chunk_task_name is not None:
            _sync_device(self.device)
            self.training_step_tracker.stop_task(task_name=self._chunk_task_name)
            self._chunk_open = False
            self._chunk_task_name = None

        if self._train_start_ns is not None:
            _sync_device(self.device)
            self._train_duration_ns = time.perf_counter_ns() - self._train_start_ns
        self._train_start_ns = None

        self.total_training_tracker.stop()
        self.training_step_tracker.stop()

    def start_step(self) -> None:
        self.iteration += 1
        if (self.iteration - 1) % self.step_interval == 0:
            _sync_device(self.device)
            first = self.iteration
            last = first + self.step_interval - 1
            self._chunk_task_name = f"Steps #{first}-#{last}"
            self.training_step_tracker.start_task(task_name=self._chunk_task_name)
            self._chunk_open = True

    def stop_step(self) -> None:
        if self.iteration % self.step_interval == 0 and self._chunk_open and self._chunk_task_name:
            _sync_device(self.device)
            self.training_step_tracker.stop_task(task_name=self._chunk_task_name)
            self._chunk_open = False
            self._chunk_task_name = None

    def start_forward(self) -> None:
        pass

    def stop_forward(self) -> None:
        pass

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_save_checkpoint(self) -> None:
        logger.warning(
            f"Method 'start_save_checkpoint' is not implemented for '{self.__class__.__name__}'."
        )

    def stop_save_checkpoint(self) -> None:
        logger.warning(
            f"Method 'stop_save_checkpoint' is not implemented for '{self.__class__.__name__}'."
        )

    def log_step(self) -> None:
        pass

    def _write_run(self) -> None:
        run_number = f"{self._file_prefix}run_{self.run_num}_"
        gpu_id = self._gpu_id
        coarse_task_path = os.path.join(
            self.output_dir,
            f"{run_number}cc_step_coarse_rank_{gpu_id}-coarse_steps.csv",
        )
        total_train_wall_s = (
            float(self._train_duration_ns) / 1.0e9
            if self._train_duration_ns is not None
            else float("nan")
        )

        n_chunks = 0
        sum_chunk_dur_s = float("nan")
        mean_chunk_dur_s = float("nan")
        per_step_from_codecarbon_s = float("nan")
        mean_full_interval_chunk_dur_s = float("nan")
        per_step_from_full_chunks_s = float("nan")

        if os.path.isfile(coarse_task_path):
            try:
                cdf = pd.read_csv(coarse_task_path)
                if "duration" in cdf.columns and len(cdf):
                    d = pd.to_numeric(cdf["duration"], errors="coerce").to_numpy(dtype=np.float64)
                    d = d[np.isfinite(d)]
                    n_chunks = int(d.size)
                    if n_chunks:
                        sum_chunk_dur_s = float(np.nansum(d))
                        mean_chunk_dur_s = float(np.nanmean(d))
                        if self.iteration > 0:
                            per_step_from_codecarbon_s = sum_chunk_dur_s / float(self.iteration)
                        full_mask = np.ones_like(d, dtype=bool)
                        if self._partial_final_chunk and n_chunks >= 1:
                            full_mask[-1] = False
                        if full_mask.any():
                            mean_full_interval_chunk_dur_s = float(np.mean(d[full_mask]))
                            per_step_from_full_chunks_s = (
                                mean_full_interval_chunk_dur_s / float(self.step_interval)
                            )
            except OSError as e:
                logger.warning("Could not read %s: %s", coarse_task_path, e)

        summary_path = os.path.join(
            self.output_dir,
            f"{run_number}cc_coarse_run_summary_rank_{gpu_id}.csv",
        )
        row = {
            "trainer_stats": "codecarbon_coarse",
            "run_num": self.run_num,
            "gpu_id": gpu_id,
            "step_interval": self.step_interval,
            "training_steps": self.iteration,
            "partial_final_chunk": int(bool(self._partial_final_chunk)),
            "n_coarse_chunks": n_chunks,
            "total_train_wall_s": total_train_wall_s,
            "codecarbon_sum_chunk_duration_s": sum_chunk_dur_s,
            "codecarbon_mean_chunk_duration_s": mean_chunk_dur_s,
            "codecarbon_per_step_wall_est_s": per_step_from_codecarbon_s,
            "codecarbon_per_step_wall_est_from_full_chunks_s": per_step_from_full_chunks_s,
        }
        pd.DataFrame([row]).to_csv(summary_path, index=False)
        logger.info(
            "CodeCarbon coarse run summary: wall_total_s=%.6f est_per_step_s=%s → %s",
            total_train_wall_s,
            per_step_from_codecarbon_s,
            summary_path,
        )

    def log_stats(self) -> None:
        self._write_run()
        run_number = f"{self._file_prefix}run_{self.run_num}_"
        gpu_id = self.device.index
        losses_dir = os.path.join(self.output_dir, "losses")
        os.makedirs(losses_dir, exist_ok=True)
        df = pd.DataFrame([[x["task_name"], x["loss"].item()] for x in self.losses])
        save_file_path = os.path.join(losses_dir, f"{run_number}cc_loss_rank_{gpu_id}.csv")
        df.to_csv(save_file_path, index=False)
        logger.info(
            "CODECARBON COARSE LOSS LOGGING: Rank %s - Run %s - Losses saved to %s",
            gpu_id,
            self.run_num,
            save_file_path,
        )

    def log_loss(self, loss: torch.Tensor) -> None:
        self.losses.append(
            {
                "task_name": f"Step #{self.iteration}",
                "loss": loss.to(torch.device("cpu"), non_blocking=True),
            }
        )
