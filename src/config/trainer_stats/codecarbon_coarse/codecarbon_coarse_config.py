from src.config.util.base_config import _Arg, _BaseConfig

config_name = "codecarbon_coarse"


class TrainerStatsConfig(_BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(
            type=int,
            help="Run number used in CodeCarbon output filenames.",
            default=0,
        )
        self._arg_project_name = _Arg(
            type=str,
            help="CodeCarbon project_name.",
            default="energy-efficiency",
        )
        self._arg_output_dir = _Arg(
            type=str,
            help="Directory for emitted CSVs.",
            default=".",
        )
        self._arg_step_interval = _Arg(
            type=int,
            help=(
                "Start/stop one CodeCarbon step-task every N training steps (no substep tasks). "
                "Larger N gives counters longer windows between updates. Use 1 for per-step tasks."
            ),
            default=10,
        )
