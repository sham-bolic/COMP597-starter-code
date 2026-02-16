from src.config.util.base_config import _Arg, _BaseConfig

config_name = "resource_util"

class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(type=str, help="Directory where resource utilization CSV will be saved.", default=".")
        self._arg_output_file = _Arg(type=str, help="Output CSV filename (within output_dir).", default="resource_util.csv")
