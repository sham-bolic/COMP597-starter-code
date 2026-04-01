from src.config.util.base_config import _Arg, _BaseConfig

config_name = "codecarbon_e2e"


class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(
            type=int,
            help="Run number used in output filenames (same convention as codecarbon).",
            default=0,
        )
        self._arg_project_name = _Arg(
            type=str,
            help="CodeCarbon project_name passed to OfflineEmissionsTracker.",
            default="energy-efficiency",
        )
        self._arg_output_dir = _Arg(
            type=str,
            help="Directory for the emitted CSV (created if missing).",
            default=".",
        )
