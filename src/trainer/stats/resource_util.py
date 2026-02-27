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
    substep_output_file = "resource_util_substeps.csv"
    ru_config = getattr(conf.trainer_stats_configs, "resource_util", None)
    if ru_config is not None:
        output_path = getattr(ru_config, "output_dir", ".")
        output_file = getattr(ru_config, "output_file", "resource_util.csv")
        substep_output_file = getattr(ru_config, "substep_output_file", "resource_util_substeps.csv")
    csv_path = os.path.join(output_path, output_file)
    substep_csv_path = os.path.join(output_path, substep_output_file)
    return ResourceUtilStats(device=device, csv_path=csv_path, substep_csv_path=substep_csv_path)

class ResourceUtilStats(simple.SimpleTrainerStats):
    # Disable tqdm to avoid terminal conflicts (metrics are written to CSV)
    SUPPRESS_PROGRESS_BAR = True

    def __init__(self, device, csv_path="resource_util.csv", substep_csv_path=None):
        super().__init__(device)
        self.csv_path = csv_path
        self.substep_csv_path = substep_csv_path or csv_path.replace(".csv", "_substeps.csv")
        self._rows = []
        self._substep_rows = []

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
        io = self.process.io_counters()
        self.io_read_start = io.read_bytes
        self.io_write_start = io.write_bytes
        output_dir = os.path.dirname(self.csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._rows = []
        self._substep_rows = []

    def _record_substep(self, phase: str):
        """Record resource stats at end of a phase (forward/backward/optimizer)."""
        if self.device.type != "cuda":
            return
        torch.cuda.synchronize(self.device)
        step_num = int(self.step_stats.stat.average.n) + 1  # current step (1-based)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        gpu_util = util.gpu
        gpu_mem_gb = mem_info.used / 1e9
        cpu_util = self.process.cpu_percent()
        cpu_mem_gb = self.process.memory_info().rss / 1e9
        ram_gb = psutil.virtual_memory().used / 1e9
        io = self.process.io_counters()
        io_read = (io.read_bytes - self.io_read_start) / 1e9
        io_write = (io.write_bytes - self.io_write_start) / 1e9
        self._substep_rows.append([
            step_num,
            phase,
            gpu_util,
            cpu_util,
            gpu_mem_gb,
            cpu_mem_gb,
            ram_gb,
            io_read,
            io_write,
        ])

    def start_step(self):
        self.step_stats.start()

    def stop_step(self):
        self.step_stats.stop()
        step_num = int(self.step_stats.stat.average.n)
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

    def start_forward(self):
        super().start_forward()

    def stop_forward(self):
        self._record_substep("forward")
        super().stop_forward()

    def start_backward(self):
        super().start_backward()

    def stop_backward(self):
        self._record_substep("backward")
        super().stop_backward()

    def start_optimizer_step(self):
        super().start_optimizer_step()

    def stop_optimizer_step(self):
        self._record_substep("optimizer")
        super().stop_optimizer_step()

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
        print("###############   GPU UTILIZATION (%)   ###############")
        self._print_quantiles("GPU Util", self.gpu_util_stats, divisor=1.0, unit="%")
        print("###############   CPU UTILIZATION (%)   ###############")
        self._print_quantiles("CPU Util", self.cpu_util_stats, divisor=1.0, unit="%")
        print("###############   GPU MEMORY USAGE   ###############")
        self._print_quantiles("GPU Mem Usage", self.gpu_mem_usage_stats, divisor=1e9, unit="GB")
        print("###############   CPU MEMORY (RSS)   ###############")
        self._print_quantiles("CPU Mem", self.cpu_mem_stats, divisor=1e9, unit="GB")
        print("###############   RAM USAGE (SYSTEM)   ###############")
        self._print_quantiles("RAM Used", self.ram_usage_stats, divisor=1e9, unit="GB")
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

        if self._substep_rows:
            with open(self.substep_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "phase", "gpu_util", "cpu_util", "gpu_mem_gb", "cpu_mem_gb", "ram_gb",
                    "io_read_gb", "io_write_gb"
                ])
                writer.writerows(self._substep_rows)
            logger.info(f"Resource utilization substeps saved to {self.substep_csv_path}")

    def log_step(self):
        # Per-step metrics are written to CSV in stop_step(); no console output
        # to avoid interfering with tqdm progress bar.
        pass