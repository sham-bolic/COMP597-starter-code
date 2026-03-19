from src.config.util.base_config import _Arg, _BaseConfig

config_name = "synthetic_whisper"

class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_data_path = _Arg(type=str, help="Path to the synthetic whisper data file.", default="")
        self._arg_num_labels = _Arg(type=int, help="Number of classes for audio classification.", default=10)
        self._arg_force_regenerate = _Arg(type=int, help="If 1, regenerate data even if file exists (overwrites cache).", default=0)
        self._arg_lazy = _Arg(type=int, help="If 1, generate samples on-demand (low memory, works with workers).", default=1)
        self._arg_n_samples = _Arg(type=int, help="Number of unique samples (lazy) or samples to generate (cached).", default=5500)
        self._arg_repeat = _Arg(type=int, help="Repeat dataset N times for lazy mode (effective length = n_samples * repeat).", default=1)
        self._arg_num_workers = _Arg(type=int, help="DataLoader num_workers (0=main process only; 2-4 for experiments to increase GPU utilization).", default=0)