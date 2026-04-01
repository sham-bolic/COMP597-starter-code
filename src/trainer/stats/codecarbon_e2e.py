"""One-shot CodeCarbon energy: a single OfflineEmissionsTracker for the full train loop.

Unlike ``codecarbon.CodeCarbonStats``, this does **not** start per-step or per-substep
task trackers and does not call into CodeCarbon on every forward/backward/optimizer
boundary (and does not add CUDA synchronizations there). Use it as a baseline to
compare against full ``--trainer_stats codecarbon`` runs when estimating
instrumentation overhead.

Output is one CSV per run, analogous to the full-run file from CodeCarbon stats but
with a distinct basename so runs are not confused:

  ``{file_prefix}run_{run_num}_cc_e2e_full_rank_{gpu_id}.csv``

The columns are produced by CodeCarbon's ``SimpleFileOutput`` (same as ``cc_full``).
"""

from __future__ import annotations

import logging
import os

import codecarbon
import codecarbon.core.cpu
import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as stats_utils
import torch
from codecarbon import OfflineEmissionsTracker

from src.trainer.stats.codecarbon import SimpleFileOutput

logger = logging.getLogger(__name__)

codecarbon.core.cpu.is_psutil_available = lambda: False

trainer_stats_name = "codecarbon_e2e"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning(
            "No device provided to codecarbon_e2e trainer stats. Using default PyTorch device"
        )
        device = torch.get_default_device()
    fp = stats_utils.trainer_stats_file_prefix(conf)
    cfg = conf.trainer_stats_configs.codecarbon_e2e
    return CodeCarbonE2EStats(
        device,
        cfg.run_num,
        cfg.project_name,
        cfg.output_dir,
        file_prefix=fp,
    )


class CodeCarbonE2EStats(base.TrainerStats):
    """Single full-training emission estimate via CodeCarbon (minimal API surface)."""

    def __init__(
        self,
        device: torch.device,
        run_num: int,
        project_name: str,
        output_dir: str,
        file_prefix: str = "",
    ) -> None:
        self.device = device
        self.run_num = run_num
        self.project_name = project_name
        self.output_dir = output_dir
        self._file_prefix = file_prefix

        gpu_id = self.device.index
        run_number = f"{file_prefix}run_{run_num}_"
        os.makedirs(self.output_dir, exist_ok=True)

        self._tracker = OfflineEmissionsTracker(
            project_name=project_name,
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[
                SimpleFileOutput(
                    output_file_name=f"{run_number}cc_e2e_full_rank_{gpu_id}.csv",
                    output_dir=output_dir,
                )
            ],
            allow_multiple_runs=True,
            log_level="warning",
            gpu_ids=[gpu_id],
        )

    def start_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self._tracker.start()

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self._tracker.stop()

    def start_step(self) -> None:
        pass

    def stop_step(self) -> None:
        pass

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
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        pass
