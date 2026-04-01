import csv
import logging
import os
import time
from pathlib import Path

import pynvml
import psutil
import torch
import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils
import src.trainer.stats.simple as simple

logger = logging.getLogger(__name__)

trainer_stats_name = "resource_util"

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to resource utils trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    output_path = "."
    output_file = "resource_util.csv"
    ru_config = getattr(conf.trainer_stats_configs, "resource_util", None)
    if ru_config is not None:
        output_path = getattr(ru_config, "output_dir", ".")
        output_file = getattr(ru_config, "output_file", "resource_util.csv")
    output_file = utils.apply_trainer_stats_file_prefix_to_basename(output_file, conf)
    csv_path = os.path.join(output_path, output_file)
    duration_name = utils.apply_trainer_stats_file_prefix_to_basename(
        utils.train_duration_basename("resource_util_train_duration"), conf
    )
    duration_path = Path(output_path) / duration_name
    return ResourceUtilStats(device=device, csv_path=csv_path, duration_path=duration_path)

class ResourceUtilStats(simple.SimpleTrainerStats):
    """Per-step resource CSV plus full-loop wall time (same timing model as ``noop``).

    Writes ``duration_ms`` to ``duration_path`` (``resource_util_train_duration*.txt``
    next to the CSV; ``memory_`` / ``RUN_REPEAT_INDEX`` suffixes match noop’s rules).
    """

    # Disable tqdm to avoid terminal conflicts (metrics are written to CSV)
    SUPPRESS_PROGRESS_BAR = True

    def __init__(self, device, csv_path="resource_util.csv", duration_path=None):
        super().__init__(device)
        self.csv_path = csv_path
        self._duration_path = Path(duration_path) if duration_path is not None else None
        self._rows = []
        self._train_start_ns = None
        self._train_duration_ns = None

        pynvml.nvmlInit()
        gpu_index = device.index if device.index is not None else 0
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        self.process = psutil.Process()

        self.gpu_util_stats = utils.RunningStat()       # GPU utilization %
        self.gpu_mem_usage_stats = utils.RunningStat()   # GPU memory usage (from NVML)
        self.cpu_util_stats = utils.RunningStat()        # CPU utilization % (this process)
        self.cpu_mem_stats = utils.RunningStat()        # Process RAM (RSS)
        self.ram_usage_stats = utils.RunningStat()      # System RAM used (whole machine)

        self.io_read_start = 0
        self.io_write_start = 0
        self.total_io_read = 0
        self.total_io_write = 0

    def start_train(self):
        self._train_duration_ns = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._train_start_ns = time.perf_counter_ns()
        io = self.process.io_counters()
        self.io_read_start = io.read_bytes
        self.io_write_start = io.write_bytes
        output_dir = os.path.dirname(self.csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._rows = []

    def start_step(self):
        self.step_stats.start()

    def stop_step(self):
        self.step_stats.stop()
        step_num = int(self.step_stats.stat.average.n)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        self.gpu_util_stats.update(util.gpu)
        self.gpu_mem_usage_stats.update(mem_info.used)
        self.cpu_util_stats.update(self.process.cpu_percent())
        self.cpu_mem_stats.update(self.process.memory_info().rss)
        self.ram_usage_stats.update(psutil.virtual_memory().used)

        io = self.process.io_counters()
        io_read = (io.read_bytes - self.io_read_start) / 1e9
        io_write = (io.write_bytes - self.io_write_start) / 1e9

        self._rows.append([
            step_num,
            self.gpu_util_stats.get_last(),
            self.cpu_util_stats.get_last(),
            self.gpu_mem_usage_stats.get_last() / 1e9,
            self.cpu_mem_stats.get_last() / 1e9,
            self.ram_usage_stats.get_last() / 1e9,
            io_read,
            io_write,
        ])

    def stop_train(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self._train_start_ns is not None:
            self._train_duration_ns = time.perf_counter_ns() - self._train_start_ns
        self._train_start_ns = None
        io = self.process.io_counters()
        self.total_io_read = io.read_bytes - self.io_read_start
        self.total_io_write = io.write_bytes - self.io_write_start

    def log_stats(self):
        if self._rows:
            gpu_util_mean = sum(r[1] for r in self._rows) / len(self._rows)
            cpu_util_mean = sum(r[2] for r in self._rows) / len(self._rows)
            gpu_mem_mean = sum(r[3] for r in self._rows) / len(self._rows)
            cpu_mem_mean = sum(r[4] for r in self._rows) / len(self._rows)
            ram_mean = sum(r[5] for r in self._rows) / len(self._rows)
            print("###############   RESOURCE UTILIZATION (mean)   ###############")
            print(f"GPU util: {gpu_util_mean:.1f}%  CPU util: {cpu_util_mean:.1f}%")
            print(f"GPU mem: {gpu_mem_mean:.4f} GB  CPU mem: {cpu_mem_mean:.4f} GB  RAM: {ram_mean:.4f} GB")
        print("###############   I/O TOTALS   ###############")
        print(f"Total I/O Read: {self.total_io_read / 1e9:.2f} GB")
        print(f"Total I/O Write: {self.total_io_write / 1e9:.2f} GB")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "gpu_util", "cpu_util", "gpu_mem_gb", "cpu_mem_gb", "ram_gb",
                "io_read_gb", "io_write_gb"
            ])
            writer.writerows(self._rows)
        logger.info(f"Resource utilization saved to {self.csv_path}")
        d = self._train_duration_ns
        path = self._duration_path
        if d is not None and path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            ms = d / 1e6
            with path.open("w", encoding="utf-8") as f:
                f.write(f"duration_ms {ms}\n")
            logger.info(f"Train duration saved to {path}")

    def log_step(self):
        # Per-step metrics are written to CSV in stop_step(); no console output
        # to avoid interfering with tqdm progress bar.
        pass
