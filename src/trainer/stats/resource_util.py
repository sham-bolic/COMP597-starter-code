import logging
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
    return ResourceUtilStats(device=device)

class ResourceUtilStats(simple.SimpleTrainerStats):

    def __init__(self, device):
        super().__init__(device)

        pynvml.nvmlInit()
        gpu_index = device.index if device.index is not None else 0
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        self.process = psutil.Process()

        self.gpu_util_stats = utils.RunningStat()       # GPU utilization %
        self.gpu_mem_alloc_stats = utils.RunningStat()   # GPU memory allocated
        self.gpu_mem_reserved_stats = utils.RunningStat() # GPU memory reserved
        self.cpu_mem_stats = utils.RunningStat()          # CPU RSS

        self.io_read_start = 0
        self.io_write_start = 0
        self.total_io_read = 0
        self.total_io_write = 0

    def start_train(self):
        io = self.process.io_counters()
        self.io_read_start = io.read_bytes
        self.io_write_start = io.write_bytes

    def stop_step(self):
        super().stop_step()
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        self.gpu_util_stats.update(util.gpu)
        self.gpu_mem_alloc_stats.update(torch.cuda.memory_allocated(self.device))
        self.gpu_mem_reserved_stats.update(torch.cuda.memory_reserved(self.device))
        self.cpu_mem_stats.update(self.process.memory_info().rss)

    def stop_train(self):
        io = self.process.io_counters()
        self.total_io_read = io.read_bytes - self.io_read_start
        self.total_io_write = io.write_bytes - self.io_write_start

    def log_stats(self):
        super().log_stats()
        print("###############   RESOURCE UTIL   ###############")
        print(f"GPU Utilization avg: {self.gpu_util_stats.get_average():.1f}%")
        print(f"GPU Memory Allocated avg: {self.gpu_mem_alloc_stats.get_average() / 1e9:.2f} GB")
        print(f"GPU Memory Reserved avg: {self.gpu_mem_reserved_stats.get_average() / 1e9:.2f} GB")
        print(f"CPU Memory (RSS) avg: {self.cpu_mem_stats.get_average() / 1e9:.2f} GB")
        print(f"Total I/O Read: {self.total_io_read / 1e9:.2f} GB")
        print(f"Total I/O Write: {self.total_io_write / 1e9:.2f} GB")