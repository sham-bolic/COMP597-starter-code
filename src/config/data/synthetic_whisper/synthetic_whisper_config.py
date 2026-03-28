from src.config.util.base_config import _Arg, _BaseConfig

config_name = "synthetic_whisper"

class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_data_path = _Arg(type=str, help="Path to the synthetic whisper data file.", default="")
        self._arg_num_labels = _Arg(type=int, help="Number of classes for audio classification.", default=10)
        self._arg_force_regenerate = _Arg(type=int, help="If 1, regenerate data even if file exists (overwrites cache).", default=0)
        self._arg_memory_only = _Arg(type=int, help="If 1, build n_samples in RAM only (no .pt cache); indexes cycle with repeat like milabench.", default=1)
        self._arg_n_samples = _Arg(type=int, help="Number of unique samples (memory-only) or samples to generate / load from cache.", default=500)
        self._arg_repeat = _Arg(type=int, help="Repeat dataset N times (effective length = n_samples * repeat).", default=1)
        self._arg_num_workers = _Arg(type=int, help="DataLoader num_workers (0=main process only; 2-4 for experiments to increase GPU utilization).", default=0)