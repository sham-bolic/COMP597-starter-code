from src.config.util.base_config import _Arg, _BaseConfig

config_name = "synthetic_whisper"

_DATA_TYPE_CHOICES = ("chunks", "shard", "memmap", "memory")


class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_data_path = _Arg(type=str, help="Path to the synthetic whisper data file.", default="")
        self._arg_num_unique_samples = _Arg(
            type=int,
            help="Unique synthetic examples for on-disk backends. Ignored when memory_only=1 (pool size is --batch_size).",
            default=16000,
        )
        self._arg_num_labels = _Arg(type=int, help="Number of classes for audio classification.", default=10)
        self._arg_force_regenerate = _Arg(type=int, help="If 1, regenerate data even if file exists (overwrites cache).", default=0)
        self._arg_memory_only = _Arg(
            type=int,
            help="If 1, force data_type=memory (RAM only). Holds batch_size unique samples (not num_unique_samples).",
            default=0,
        )
        self._arg_data_type = _Arg(
            type=str,
            choices=_DATA_TYPE_CHOICES,
            help="Storage backend: chunks (chunked .pt), shard, memmap, memory.",
            default="chunks",
        )
        self._arg_chunk_size = _Arg(
            type=int,
            help="Samples per chunk file when data_type=chunks (default 400). Ignored for shard/memmap/memory.",
            default=400,
        )
        self._arg_num_shards = _Arg(
            type=int,
            help="Number of shard files when data_type=shard (samples split across shards).",
            default=4,
        )
        self._arg_repeat = _Arg(
            type=int,
            help="Repeat dataset N times. Len = num_unique_samples × repeat (disk) or batch_size × repeat (memory_only).",
            default=1,
        )
        self._arg_num_workers = _Arg(type=int, help="DataLoader num_workers (0=main process only; 2-4 for experiments to increase GPU utilization).", default=0)
