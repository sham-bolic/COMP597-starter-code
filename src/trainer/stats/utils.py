from __future__ import annotations

import logging
import os
import pynvml
import time
import torch
import src.config as config

logger = logging.getLogger(__name__)

_TRAINER_STATS_AUTO_DISCOVERY_IGNORE=True

class RunningAverage:
    """Implements a running average.
    
    Attributes
    ----------
    average : float
        Current average.
    n : int
        Number of samples.
    """

    def __init__(self) -> None:
        self.average = 0
        self.n = 0

    def update(self, value : int) -> None:
        """Update the average with a new value.
        
        This will perform an inplace update on the average.

        Parameters
        ----------
        value
            The value to include in the running average.

        """
        self.average = (self.average * self.n + value) / (self.n + 1)
        self.n += 1

    def get(self) -> float:
        """The current average.
        """
        return self.average

class RunningStat:
    """An updatable statistics.

    This is used to accumulate an integer statistics and provide a breakdown on 
    the data.

    Attributes
    ----------
    average : RunningAverage
        The running average of the statistics.
    history : List[Any]
        All the values accumulated.

    """

    # TODO Add a unit transformation parameter. Log analysis should not by default divide values by 1e6. It should be configurable.
    def __init__(self) -> None:
       self.average = RunningAverage()
       self.history = []

    def update(self, value : int) -> None:
        """Include a new value.

        This includes a new value in the history and the running average.

        Parameters
        ----------
        value
            The value to add to the statistic.

        """
        self.history.append(value)
        self.average.update(value)

    def get_average(self) -> float:
        """The current average of the statistic.
        """
        return self.average.get()

    def get_last(self) -> int:
        """The last value added to the statistic.
        """
        if len(self.history) == 0:
            return -1
        return self.history[-1]

    def log_analysis(self) -> None:
        """Logs the mean and key quantiles of accumulated data.
        """
        data = torch.tensor(self.history)
        data = data.to(torch.float)
        print(f"mean   : {data.mean() / 1000000 : .4f}")
        print(f"q0.001 : {data.quantile(q=torch.tensor(0.001), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.01  : {data.quantile(q=torch.tensor(0.010), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.1   : {data.quantile(q=torch.tensor(0.100), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.25  : {data.quantile(q=torch.tensor(0.250), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.5   : {data.quantile(q=torch.tensor(0.500), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.75  : {data.quantile(q=torch.tensor(0.750), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.9   : {data.quantile(q=torch.tensor(0.900), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.99  : {data.quantile(q=torch.tensor(0.990), interpolation='nearest') / 1000000 : .4f}")
        print(f"q0.999 : {data.quantile(q=torch.tensor(0.999), interpolation='nearest') / 1000000 : .4f}")

class RunningTimer:
    """A reusable timer.

    Provides a timer that can be reused. It keeps track of all the time 
    measurements performed. Measurements are in nanoseconds as provided by 
    Python's native `time.perf_counter_ns()`.

    Attributes
    ----------
    stat : RunningStat
        The statistic object containing all the measurements done with this timer.
    stats_ts : int
        The timestamp measured by `start`.

    Notes
    -----
        This timer does not support recursive starts. It should be stopped 
        before being started again. There is not verification of proper usage, 
        it is the programmer's task to ensure proper usage.

    """

    def __init__(self) -> None:
       self.stat = RunningStat()
       self.start_ts = 0

    def start(self) -> None:
        """Start the timer.
        
        Starts the timer. The timestamp is stored in the attribute `start_ts`.

        """
        self.start_ts = time.perf_counter_ns()

    def stop(self) -> None:
        """Stop the timer.

        Stops the timer. It records the time different between the timestamp of 
        when the timer was started and the current timestamp. The measurement 
        is added to the statistics.
        
        """
        self.stat.update(time.perf_counter_ns() - self.start_ts)

    def get_last(self) -> int:
        """The last time measurement.
        """
        return self.stat.get_last()

    def get_average(self) -> float:
        """The average time measurement.
        """
        return self.stat.get_average()

    def log_analysis(self) -> None:
        """Analysis of the time measurements.

        Refer to the `log_analysis` implementation of `RunningStat` for more 
        details on the output of this method.

        """
        self.stat.log_analysis()

class RunningEnergy:
    """A reusable GPU energy counter.

    Provides an energy counter for GPUs that can be reused. It keeps track of 
    all the energy consumption measurements performed. Measurements are in 
    millijoules.

    Parameters
    ----------
    gpu_index
        The index of the GPU for which to measure the energy consumption.

    Attributes
    ----------
    stat : RunningStat
        The statistic object containing all the measurements done with this 
        counter.
    start_energy : int
        The energy measured when the counter was started.
    gpu_index : int
        The GPU index as provided to the constructor.
    handle : pynvml._Pointer[struct_c_nvmlDevice_t]
        The NVML handle of the GPU tracked by this counter..

    Notes
    -----
        This counter does not support recursive starts. It should be stopped 
        before being started again. There is not verification of proper usage, 
        it is the programmer's task to ensure proper usage.

    """

    def __init__(self, gpu_index : int) -> None:
        self.stat = RunningStat()
        self.start_energy = 0
        if gpu_index is None:
            logger.warning("GPU index not provided, defaulting to 0.")
            gpu_index = 0
        self.gpu_index = gpu_index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    def _get_energy(self) -> int:
        """The current Energy consumption of the GPU.

        This method returns the total energy consumption in millijoules that 
        the GPU has consumed since the last the driver was reloaded as 
        specified by the NVML library.
        
        """
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def start(self) -> None:
        """Start the counter.

        It records the energy consumption *timestamp* of the GPU.

        """
        self.start_energy = self._get_energy()

    def stop(self) -> None:
        """Stops the counter.

        The recorded energy measurement is the different between the current 
        GPU energy consumption and the energy consumption when the counter was 
        started. This yields how much energy was consumed between the calls to 
        `start` and `stop`.

        """
        self.stat.update(self._get_energy() - self.start_energy)

    def get_last(self) -> int: 
        """The last energy measurement.
        """
        return self.stat.get_last()

    def get_average(self) -> float:
        """The average energy consumptions so far.
        """
        return self.stat.get_average()

    def log_analysis(self) -> None:
        """Analysis of the energy measurements.

        Refer to the `log_analysis` implementation of `RunningStat` for more 
        details on the output of this method.

        """
        self.stat.log_analysis()


def trainer_stats_file_prefix(conf: config.Config) -> str:
    """Return ``memory_`` when synthetic_whisper effective storage is ``memory``, else ``""``."""
    from src.data.synthetic_whisper.data import effective_synthetic_whisper_data_type

    sc = getattr(conf.data_configs, "synthetic_whisper", None)
    if sc is None:
        return ""
    try:
        if effective_synthetic_whisper_data_type(sc) != "memory":
            return ""
    except (TypeError, ValueError):
        return ""
    return "memory_"


def apply_trainer_stats_file_prefix_to_basename(basename: str, conf: config.Config) -> str:
    """Prefix ``memory_`` to a filename basename when in synthetic in-memory data mode."""
    pfx = trainer_stats_file_prefix(conf)
    if not pfx or not basename:
        return basename
    if basename.startswith(pfx):
        return basename
    return f"{pfx}{basename}"


def train_duration_basename(stem: str = "baseline_train_duration") -> str:
    """Basename for full-loop duration files (``RUN_REPEAT_INDEX`` → ``_run_<n>`` suffix).

    Use stem ``baseline_train_duration`` (noop) or ``resource_util_train_duration`` (resource_util).
    """
    idx = os.environ.get("RUN_REPEAT_INDEX", "").strip()
    if idx:
        return f"{stem}_run_{idx}.txt"
    return f"{stem}.txt"
