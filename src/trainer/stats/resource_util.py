import csv
import logging
import os
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
    csv_path = os.path.join(output_path, output_file)
    return ResourceUtilStats(device=device, csv_path=csv_path)

class ResourceUtilStats(simple.SimpleTrainerStats):

    def __init__(self, device, csv_path="resource_util.csv"):
        super().__init__(device)
        self.csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None

        pynvml.nvmlInit()
        gpu_index = device.index if device.index is not None else 0
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        self.process = psutil.Process()

        self.gpu_util_stats = utils.RunningStat()       # GPU utilization %
        self.gpu_mem_usage_stats = utils.RunningStat()   # consumed/allocated % (this process)
        self.cpu_util_stats = utils.RunningStat()        # CPU utilization % (this process)
        self.cpu_mem_stats = utils.RunningStat()        # Process RAM (RSS)
        self.ram_usage_stats = utils.RunningStat()      # System RAM used (whole machine)

        self.io_read_start = 0
        self.io_write_start = 0
        self.total_io_read = 0
        self.total_io_write = 0

    def start_train(self):
        io = self.process.io_counters()
        self.io_read_start = io.read_bytes
        self.io_write_start = io.write_bytes
        output_dir = os.path.dirname(self.csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "step", "phase", "gpu_util", "cpu_util", "gpu_mem_pct", "cpu_mem_gb", "ram_gb",
            "io_read_gb", "io_write_gb"
        ])

    def _record_phase_metrics(self, phase: str) -> None:
        """Capture and write resource metrics for a training phase."""
        step_num = int(self.step_stats.stat.average.n) + 1  # 1-indexed, before stop_step
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        io = self.process.io_counters()
        io_read = (io.read_bytes - self.io_read_start) / 1e9
        io_write = (io.write_bytes - self.io_write_start) / 1e9
        if self._csv_writer is not None:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            gpu_mem_pct = 100.0 * allocated / reserved if reserved > 0 else 0
            self._csv_writer.writerow([
                step_num,
                phase,
                util.gpu,
                self.process.cpu_percent(),
                gpu_mem_pct,
                self.process.memory_info().rss / 1e9,
                psutil.virtual_memory().used / 1e9,
                io_read,
                io_write,
            ])
            self._csv_file.flush()

    def stop_forward(self):
        super().stop_forward()
        self._record_phase_metrics("forward")

    def stop_backward(self):
        super().stop_backward()
        self._record_phase_metrics("backward")

    def stop_optimizer_step(self):
        super().stop_optimizer_step()
        self._record_phase_metrics("optimizer")

    def stop_step(self):
        super().stop_step()
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        self.gpu_util_stats.update(util.gpu)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        gpu_mem_pct = 100.0 * allocated / reserved if reserved > 0 else 0
        self.gpu_mem_usage_stats.update(gpu_mem_pct)
        self.cpu_util_stats.update(self.process.cpu_percent())
        self.cpu_mem_stats.update(self.process.memory_info().rss)
        self.ram_usage_stats.update(psutil.virtual_memory().used)

    def stop_train(self):
        io = self.process.io_counters()
        self.total_io_read = io.read_bytes - self.io_read_start
        self.total_io_write = io.write_bytes - self.io_write_start

    def _print_quantiles(self, name, stat, divisor=1.0, unit=""):
        data = torch.tensor(stat.history, dtype=torch.float)
        quantiles = [0.25, 0.5, 0.75, 0.999]
        print(f"mean   : {data.mean() / divisor:.4f} {unit}")
        for q in quantiles:
            val = data.quantile(q=torch.tensor(q), interpolation='nearest')
            print(f"q{q:<5} : {val / divisor:.4f} {unit}")

    def log_stats(self):
        super().log_stats()
        print("###############   GPU UTILIZATION (%)   ###############")
        self._print_quantiles("GPU Util", self.gpu_util_stats, divisor=1.0, unit="%")
        print("###############   CPU UTILIZATION (%)   ###############")
        self._print_quantiles("CPU Util", self.cpu_util_stats, divisor=1.0, unit="%")
        print("###############   GPU MEMORY (CONSUMED/ALLOCATED %)   ###############")
        self._print_quantiles("GPU Mem %", self.gpu_mem_usage_stats, divisor=1.0, unit="%")
        print("###############   CPU MEMORY (RSS)   ###############")
        self._print_quantiles("CPU Mem", self.cpu_mem_stats, divisor=1e9, unit="GB")
        print("###############   RAM USAGE (SYSTEM)   ###############")
        self._print_quantiles("RAM Used", self.ram_usage_stats, divisor=1e9, unit="GB")
        print("###############   I/O TOTALS   ###############")
        print(f"Total I/O Read: {self.total_io_read / 1e9:.2f} GB")
        print(f"Total I/O Write: {self.total_io_write / 1e9:.2f} GB")
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
            logger.info(f"Resource utilization saved to {self.csv_path}")

    def log_step(self):
        # Per-step metrics are written to CSV in stop_step(); no console output
        # to avoid interfering with tqdm progress bar.
        pass