from src.config.util.base_config import _Arg, _BaseConfig

config_name = "synthetic_whisper"

class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_data_path = _Arg(type=str, help="Path to the synthetic whisper data file.", default="")