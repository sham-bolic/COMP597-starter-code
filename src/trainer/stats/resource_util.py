import csv
import logging
import math
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

    Tracks **two pairs** of cumulative I/O counters from ``/proc/pid/io`` (via ``psutil``
    ``io_counters``), covering the trainer process plus all descendants (``DataLoader`` workers):

    * ``io_read_logical_gb`` / ``io_write_logical_gb`` — ``rchar`` / ``wchar``: every byte
      returned by ``read()`` / ``write()`` syscalls, **including page-cache hits**.
    * ``io_read_gb`` / ``io_write_gb`` — ``read_bytes`` / ``write_bytes``: bytes actually
      fetched from / written to the **block device**; page-cache hits are **not** counted.

    Comparing the two reveals how much data is served from cache vs disk.

    Writes ``duration_ms`` to ``duration_path`` (``resource_util_train_duration*.txt``
    next to the CSV; ``memory_`` / ``RUN_REPEAT_INDEX`` suffixes match noop's rules).
    """

    SUPPRESS_PROGRESS_BAR = True

    def __init__(self, device, csv_path="resource_util.csv", duration_path=None):
        super().__init__(device)
        self.csv_path = csv_path
        self._duration_path = Path(duration_path) if duration_path is not None else None
        self._rows = []
        self._train_start_ns = None
        self._train_duration_ns = None

        self.gpu_handle = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                gpu_index = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
            except Exception as e:
                logger.warning("resource_util: NVML init failed (%s); gpu_util will be NaN", e)

        self.process = psutil.Process()

        self.gpu_util_stats = utils.RunningStat()
        self.cpu_util_stats = utils.RunningStat()
        self.cpu_mem_stats = utils.RunningStat()
        self.ram_usage_stats = utils.RunningStat()

        # Per-pid last-seen counters: (read_chars, write_chars, read_bytes, write_bytes)
        self._io_pid_last: dict[int, tuple[int, int, int, int]] = {}
        self._io_logical_read_cumulative = 0
        self._io_logical_write_cumulative = 0
        self._io_disk_read_cumulative = 0
        self._io_disk_write_cumulative = 0
        self.total_io_logical_read = 0
        self.total_io_logical_write = 0
        self.total_io_disk_read = 0
        self.total_io_disk_write = 0

    def _process_tree_proc_list(self) -> list[psutil.Process]:
        """Trainer process and all recursive children (``DataLoader`` workers), deduped by pid."""
        seen: set[int] = set()
        out: list[psutil.Process] = []
        for p in (self.process, *self.process.children(recursive=True)):
            if p.pid in seen:
                continue
            seen.add(p.pid)
            out.append(p)
        return out

    @staticmethod
    def _extract_io_fields(io) -> tuple[int, int, int, int]:
        """Return (read_chars, write_chars, read_bytes, write_bytes) from io_counters."""
        rc = int(getattr(io, "read_chars", 0))
        wc = int(getattr(io, "write_chars", 0))
        rb = int(getattr(io, "read_bytes", 0))
        wb = int(getattr(io, "write_bytes", 0))
        return rc, wc, rb, wb

    def _prime_io_tree_baselines(self) -> None:
        """Snapshot ``io_counters`` for processes alive at train start (no bytes added to totals)."""
        self._io_pid_last.clear()
        for p in self._process_tree_proc_list():
            try:
                io = p.io_counters()
                self._io_pid_last[p.pid] = self._extract_io_fields(io)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logger.debug("resource_util: prime skip pid %s: %s", getattr(p, "pid", "?"), e)

    def _advance_tree_io_counters(self) -> tuple[float, float, float, float]:
        """Accumulate deltas for the process tree; return cumulative GB totals.

        Returns (logical_read_gb, logical_write_gb, disk_read_gb, disk_write_gb).
        """
        for p in self._process_tree_proc_list():
            try:
                io = p.io_counters()
                pid = p.pid
                rc, wc, rb, wb = self._extract_io_fields(io)
                if pid in self._io_pid_last:
                    prc, pwc, prb, pwb = self._io_pid_last[pid]
                    d_rc = max(0, rc - prc)
                    d_wc = max(0, wc - pwc)
                    d_rb = max(0, rb - prb)
                    d_wb = max(0, wb - pwb)
                    self._io_logical_read_cumulative += d_rc
                    self._io_logical_write_cumulative += d_wc
                    self._io_disk_read_cumulative += d_rb
                    self._io_disk_write_cumulative += d_wb
                else:
                    self._io_logical_read_cumulative += rc
                    self._io_logical_write_cumulative += wc
                    self._io_disk_read_cumulative += rb
                    self._io_disk_write_cumulative += wb
                self._io_pid_last[pid] = (rc, wc, rb, wb)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logger.debug("resource_util: skip pid %s io_counters: %s", getattr(p, "pid", "?"), e)
                continue
        return (
            self._io_logical_read_cumulative / 1e9,
            self._io_logical_write_cumulative / 1e9,
            self._io_disk_read_cumulative / 1e9,
            self._io_disk_write_cumulative / 1e9,
        )

    def start_train(self):
        self._train_duration_ns = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._train_start_ns = time.perf_counter_ns()
        self._io_logical_read_cumulative = 0
        self._io_logical_write_cumulative = 0
        self._io_disk_read_cumulative = 0
        self._io_disk_write_cumulative = 0
        self._prime_io_tree_baselines()
        output_dir = os.path.dirname(self.csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._rows = []

    def start_step(self):
        self.step_stats.start()

    def stop_step(self):
        self.step_stats.stop()
        step_num = int(self.step_stats.stat.average.n)
        gpu_u = float("nan")
        cuda_alloc_gb = float("nan")
        cuda_reserved_gb = float("nan")
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            cuda_alloc_gb = torch.cuda.memory_allocated(self.device) / 1e9
            cuda_reserved_gb = torch.cuda.memory_reserved(self.device) / 1e9
            if self.gpu_handle is not None:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_u = float(util.gpu)
        self.gpu_util_stats.update(gpu_u)
        self.cpu_util_stats.update(self.process.cpu_percent())
        self.cpu_mem_stats.update(self.process.memory_info().rss)
        self.ram_usage_stats.update(psutil.virtual_memory().used)

        lr, lw, dr, dw = self._advance_tree_io_counters()

        self._rows.append([
            step_num,
            self.gpu_util_stats.get_last(),
            self.cpu_util_stats.get_last(),
            cuda_alloc_gb,
            cuda_reserved_gb,
            self.cpu_mem_stats.get_last() / 1e9,
            self.ram_usage_stats.get_last() / 1e9,
            lr,
            lw,
            dr,
            dw,
        ])

    def stop_train(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self._train_start_ns is not None:
            self._train_duration_ns = time.perf_counter_ns() - self._train_start_ns
        self._train_start_ns = None
        self._advance_tree_io_counters()
        self.total_io_logical_read = self._io_logical_read_cumulative
        self.total_io_logical_write = self._io_logical_write_cumulative
        self.total_io_disk_read = self._io_disk_read_cumulative
        self.total_io_disk_write = self._io_disk_write_cumulative

    def log_stats(self):
        if self._rows:
            def _finite_mean_and_count(idx: int) -> tuple[float, int, int]:
                """Mean over finite samples only, with counts so gaps are visible."""
                vals = [float(r[idx]) for r in self._rows]
                finite = [x for x in vals if math.isfinite(x)]
                total = len(vals)
                mean = sum(finite) / len(finite) if finite else float("nan")
                return mean, len(finite), total

            gpu_util_mean, gpu_fin, gpu_tot = _finite_mean_and_count(1)
            cpu_util_mean, _, _ = _finite_mean_and_count(2)
            cuda_alloc_mean, ca_fin, ca_tot = _finite_mean_and_count(3)
            cuda_reserved_mean, cr_fin, cr_tot = _finite_mean_and_count(4)
            cpu_mem_mean, _, _ = _finite_mean_and_count(5)
            ram_mean, _, _ = _finite_mean_and_count(6)
            print("###############   RESOURCE UTILIZATION (mean)   ###############")
            print(f"GPU util: {gpu_util_mean:.1f}%  CPU util: {cpu_util_mean:.1f}%")
            print(
                f"CUDA mem allocated: {cuda_alloc_mean:.4f} GB  reserved: {cuda_reserved_mean:.4f} GB  "
                f"CPU mem: {cpu_mem_mean:.4f} GB  RAM: {ram_mean:.4f} GB"
            )
            cov = [
                (name, nf, nt)
                for name, nf, nt in (
                    ("gpu_util", gpu_fin, gpu_tot),
                    ("cuda_mem_allocated", ca_fin, ca_tot),
                    ("cuda_mem_reserved", cr_fin, cr_tot),
                )
                if nt and nf < nt
            ]
            if cov:
                msg = "; ".join(f"{n}: {nf}/{nt} finite steps (excluded NaN from mean)" for n, nf, nt in cov)
                print(f"resource_util: incomplete metrics — {msg}")
        print("###############   I/O TOTALS   ###############")
        print(f"Logical  (read_chars/write_chars): Read {self.total_io_logical_read / 1e9:.2f} GB  Write {self.total_io_logical_write / 1e9:.2f} GB")
        print(f"Disk     (read_bytes/write_bytes): Read {self.total_io_disk_read / 1e9:.2f} GB  Write {self.total_io_disk_write / 1e9:.2f} GB")
        print("Logical includes page-cache hits; Disk counts only block-device traffic.")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "gpu_util",
                "cpu_util",
                "cuda_mem_allocated_gb",
                "cuda_mem_reserved_gb",
                "cpu_mem_gb",
                "ram_gb",
                "io_read_logical_gb",
                "io_write_logical_gb",
                "io_read_gb",
                "io_write_gb",
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
        pass
