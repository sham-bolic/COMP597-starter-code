from src.config.util.base_config import _Arg, _BaseConfig

config_name = "phase_timing"


class TrainerStatsConfig(_BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(
            type=int,
            help="Run number used in output filenames (same convention as codecarbon).",
            default=0,
        )
        self._arg_output_dir = _Arg(
            type=str,
            help="Directory for phase timing CSVs (created if missing).",
            default=".",
        )
        self._arg_warmup_steps = _Arg(
            type=int,
            help="Exclude the first N steps (0-based) from summary mean/std only; per-step CSV keeps all steps.",
            default=0,
        )
        self._arg_measure_phase = _Arg(
            type=str,
            help=(
                "Which interval to time per step: all (default), step, forward, backward, or optimizer. "
                "Only that hook pair uses CUDA sync + perf_counter; others are skipped so overhead matches "
                "timing a single phase. Unmeasured columns are empty in the per-step CSV."
            ),
            default="all",
        )
