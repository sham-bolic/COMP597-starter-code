from src.config.util.base_config import _Arg, _BaseConfig

config_name = "synthetic_whisper"

class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_data_path = _Arg(type=str, help="Path to the synthetic whisper data file.", default="")
        self._arg_num_labels = _Arg(type=int, help="Number of classes for audio classification.", default=10)
        self._arg_force_regenerate = _Arg(type=int, help="If 1, regenerate data even if file exists (overwrites cache).", default=0)