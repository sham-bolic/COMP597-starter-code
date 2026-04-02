import gc
import os
import shutil
import torch
import torch.utils.data
import numpy as np
import src.config as config
from transformers import WhisperFeatureExtractor

data_load_name = "synthetic_whisper"

SAMPLE_RATE = 16000
CHUNK_SIZE_DEFAULT = 100
CHUNKED_FORMAT = "chunked_v1" 
SHARDED_FORMAT = "sharded_v1"
MEMMAP_FORMAT = "memmap_v1"
SINGLE_FILE_FORMAT = "single_file_v1"


def _chunks_dir_for(data_path: str) -> str:
    root, _ = os.path.splitext(data_path)
    return f"{root}_synthetic_chunks"


def _shards_dir_for(data_path: str) -> str:
    root, _ = os.path.splitext(data_path)
    return f"{root}_synthetic_shards"


def _memmap_dir_for(data_path: str) -> str:
    root, _ = os.path.splitext(data_path)
    return f"{root}_synthetic_memmap"


def effective_synthetic_whisper_data_type(sc) -> str:
    """Resolve data_type after legacy memory_only override."""
    if int(getattr(sc, "memory_only", 0)) == 1:
        return "memory"
    dt = str(getattr(sc, "data_type", "chunks")).strip().lower()
    valid = {"chunks", "shard", "memmap", "single_file", "memory"}
    if dt not in valid:
        raise ValueError(
            f"synthetic_whisper.data_type must be one of {sorted(valid)!r}, got {dt!r}"
        )
    return dt


def clear_synthetic_whisper_disk_cache(data_path: str, *, dry_run: bool = False) -> list[str]:
    """Remove manifest at ``data_path`` and sidecar dirs (chunks/shards/memmap stems).

    Same cleanup used when regenerating on-disk ``data_type`` (``force_regenerate=1``).
    Uses fixed sibling dir names from ``data_path`` (no manifest read).

    Returns paths that were removed or would be removed when ``dry_run`` is True.
    """
    removed: list[str] = []
    for d in (_chunks_dir_for(data_path), _shards_dir_for(data_path), _memmap_dir_for(data_path)):
        if not os.path.isdir(d):
            continue
        if dry_run:
            removed.append(d)
        else:
            shutil.rmtree(d)
            removed.append(d)
    if os.path.isfile(data_path):
        if dry_run:
            removed.append(data_path)
        else:
            os.remove(data_path)
            removed.append(data_path)
    return removed


def _one_sample(feature_extractor, num_labels: int, sample_rate: int = SAMPLE_RATE) -> dict:
    wav = (torch.rand(sample_rate) * 2 - 1).tolist()
    input_features = feature_extractor(
        wav,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )["input_features"][0]
    label = torch.randint(0, num_labels, ())
    return {
        "input_features": input_features,
        "labels": label,
    }


def _sample_list(n: int, num_labels: int, sample_rate: int = SAMPLE_RATE) -> list:
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    return [_one_sample(feature_extractor, num_labels, sample_rate) for _ in range(n)]


def generate_samples(n: int, data_path: str, num_labels: int, chunk_size: int) -> None:
    """Write samples in chunk files (peak RAM ≈ one chunk, not ``n``)."""
    chunk_size = max(1, int(chunk_size))
    parent = os.path.dirname(os.path.abspath(data_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    clear_synthetic_whisper_disk_cache(data_path, dry_run=False)
    chunks_dir = _chunks_dir_for(data_path)
    os.makedirs(chunks_dir, exist_ok=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    n_chunks = (n + chunk_size - 1) // chunk_size
    chunk_idx = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch = [_one_sample(feature_extractor, num_labels) for _ in range(end - start)]
        torch.save(batch, os.path.join(chunks_dir, f"chunk_{chunk_idx:04d}.pt"))
        print(
            f"synthetic_whisper: saved chunk {chunk_idx + 1}/{n_chunks} "
            f"({end}/{n} samples)",
            flush=True,
        )
        del batch
        gc.collect()
        chunk_idx += 1
    torch.save(
        {
            "format": CHUNKED_FORMAT,
            "n": n,
            "num_labels": num_labels,
            "chunk_size": chunk_size,
            "chunks_dir": os.path.abspath(chunks_dir),
        },
        data_path,
    )


def _even_shard_sizes(n: int, num_shards: int) -> list[int]:
    num_shards = max(1, min(int(num_shards), max(1, n)))
    base = n // num_shards
    rem = n % num_shards
    return [base + (1 if i < rem else 0) for i in range(num_shards)]


def generate_sharded(n: int, data_path: str, num_labels: int, num_shards: int) -> None:
    parent = os.path.dirname(os.path.abspath(data_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    clear_synthetic_whisper_disk_cache(data_path, dry_run=False)
    shards_dir = _shards_dir_for(data_path)
    os.makedirs(shards_dir, exist_ok=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    shard_sizes = _even_shard_sizes(n, num_shards)
    num_shards = len(shard_sizes)
    for sidx, sz in enumerate(shard_sizes):
        batch = [_one_sample(feature_extractor, num_labels) for _ in range(sz)]
        torch.save(batch, os.path.join(shards_dir, f"shard_{sidx:04d}.pt"))
        print(
            f"synthetic_whisper: saved shard {sidx + 1}/{num_shards} ({sz} samples)",
            flush=True,
        )
        del batch
        gc.collect()
    torch.save(
        {
            "format": SHARDED_FORMAT,
            "n": n,
            "num_labels": num_labels,
            "num_shards": num_shards,
            "shard_sizes": shard_sizes,
            "shards_dir": os.path.abspath(shards_dir),
        },
        data_path,
    )


def generate_single_file(n: int, data_path: str, num_labels: int, chunk_size: int) -> None:
    """One manifest ``.pt`` holding the full sample list; build in batches of ``chunk_size`` (peak RAM ≈ list length)."""
    chunk_size = max(1, int(chunk_size))
    parent = os.path.dirname(os.path.abspath(data_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    clear_synthetic_whisper_disk_cache(data_path, dry_run=False)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    samples: list = []
    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        batch = [_one_sample(feature_extractor, num_labels) for _ in range(end - start)]
        samples.extend(batch)
        del batch
        gc.collect()
        print(
            f"synthetic_whisper: single_file build {ci + 1}/{n_chunks} ({end}/{n} samples)",
            flush=True,
        )
    torch.save(
        {
            "format": SINGLE_FILE_FORMAT,
            "n": n,
            "num_labels": num_labels,
            "samples": samples,
        },
        data_path,
    )


def _pad_feature_stack(samples: list) -> tuple[torch.Tensor, tuple[int, int]]:
    max_m = max(int(s["input_features"].shape[0]) for s in samples)
    max_t = max(int(s["input_features"].shape[1]) for s in samples)
    out = torch.zeros(len(samples), max_m, max_t, dtype=torch.float32)
    for i, s in enumerate(samples):
        x = s["input_features"].float()
        out[i, : x.shape[0], : x.shape[1]] = x
    return out, (max_m, max_t)


def generate_memmap(n: int, data_path: str, num_labels: int) -> None:
    """Write memmap dataset; peak RAM is one sample tensor + mmap rows, not ``(n, H, W)`` Torch slab.

    Two passes with RNG checkpoint preserve the same sample sequence as the legacy
    list + ``_pad_feature_stack`` implementation.
    """
    parent = os.path.dirname(os.path.abspath(data_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    clear_synthetic_whisper_disk_cache(data_path, dry_run=False)
    memmap_dir = _memmap_dir_for(data_path)
    os.makedirs(memmap_dir, exist_ok=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    features_path = os.path.join(memmap_dir, "features.bin")
    labels_path = os.path.join(memmap_dir, "labels.bin")

    rng0 = torch.get_rng_state()
    max_m = max_t = 0
    for i in range(n):
        s = _one_sample(feature_extractor, num_labels)
        h, w = int(s["input_features"].shape[0]), int(s["input_features"].shape[1])
        max_m = max(max_m, h)
        max_t = max(max_t, w)
        del s
        if n > 0 and ((i + 1) % 1000 == 0 or i == n - 1):
            print(
                f"synthetic_whisper: memmap pass 1 (dims) {i + 1}/{n}",
                flush=True,
            )
        if (i + 1) % 500 == 0:
            gc.collect()

    torch.set_rng_state(rng0)
    feat_hw = (max_m, max_t)
    features_mm = np.memmap(features_path, dtype=np.float32, mode="w+", shape=(n, max_m, max_t))
    labels_mm = np.memmap(labels_path, dtype=np.int64, mode="w+", shape=(n,))
    try:
        for i in range(n):
            s = _one_sample(feature_extractor, num_labels)
            x = np.ascontiguousarray(s["input_features"].float().numpy(), dtype=np.float32)
            h, w = int(x.shape[0]), int(x.shape[1])
            row = features_mm[i]
            row[:] = 0.0
            row[:h, :w] = x
            labels_mm[i] = int(s["labels"])
            del s, x, row
            if n > 0 and ((i + 1) % 1000 == 0 or i == n - 1):
                print(
                    f"synthetic_whisper: memmap pass 2 (write) {i + 1}/{n}",
                    flush=True,
                )
            if (i + 1) % 500 == 0:
                gc.collect()
        features_mm.flush()
        labels_mm.flush()
    finally:
        del features_mm, labels_mm
        gc.collect()

    torch.save(
        {
            "format": MEMMAP_FORMAT,
            "n": n,
            "num_labels": num_labels,
            "memmap_dir": os.path.abspath(memmap_dir),
            "features_file": "features.bin",
            "labels_file": "labels.bin",
            "features_spatial_shape": list(feat_hw),
            "features_dtype": "float32",
            "labels_dtype": "int64",
        },
        data_path,
    )
    print(f"synthetic_whisper: wrote memmap dataset under {memmap_dir}", flush=True)


class SyntheticWhisperData(torch.utils.data.Dataset):

    def __init__(self, samples, repeat: int = 1):
        self.samples = samples
        self._n = len(samples)
        self.repeat = max(1, int(repeat))

    def __getitem__(self, i):
        return self.samples[i % self._n]

    def __len__(self):
        return self._n * self.repeat


class SyntheticWhisperChunkedData(torch.utils.data.Dataset):
    """Chunk files; loads one chunk file at a time into a small cache."""

    def __init__(self, chunks_dir: str, n: int, chunk_size: int, repeat: int = 1):
        self.chunks_dir = chunks_dir
        self._n = int(n)
        self.chunk_size = max(1, int(chunk_size))
        self.repeat = max(1, int(repeat))
        self._cache_chunk_idx: int | None = None
        self._cache_list: list | None = None

    def __getitem__(self, i):
        idx = i % self._n
        cidx = idx // self.chunk_size
        within = idx % self.chunk_size
        if self._cache_chunk_idx != cidx:
            path = os.path.join(self.chunks_dir, f"chunk_{cidx:04d}.pt")
            self._cache_list = torch.load(path, weights_only=False)
            self._cache_chunk_idx = cidx
        return self._cache_list[within]

    def __len__(self):
        return self._n * self.repeat


class SyntheticWhisperShardedData(torch.utils.data.Dataset):

    def __init__(self, shards_dir: str, n: int, shard_sizes: list[int], repeat: int = 1):
        self.shards_dir = shards_dir
        self._n = int(n)
        self.shard_sizes = [max(1, int(x)) for x in shard_sizes]
        self.repeat = max(1, int(repeat))
        self._cum: list[int] = [0]
        for sz in self.shard_sizes:
            self._cum.append(self._cum[-1] + sz)
        self._cache_shard_idx: int | None = None
        self._cache_list: list | None = None

    def __getitem__(self, i):
        idx = i % self._n
        sidx = 0
        while sidx + 1 < len(self.shard_sizes) and idx >= self._cum[sidx + 1]:
            sidx += 1
        within = idx - self._cum[sidx]
        if self._cache_shard_idx != sidx:
            path = os.path.join(self.shards_dir, f"shard_{sidx:04d}.pt")
            self._cache_list = torch.load(path, weights_only=False)
            self._cache_shard_idx = sidx
        return self._cache_list[within]

    def __len__(self):
        return self._n * self.repeat


class SyntheticWhisperMemmapData(torch.utils.data.Dataset):

    def __init__(
        self,
        features_path: str,
        labels_path: str,
        n: int,
        spatial: tuple[int, int],
        repeat: int = 1,
    ):
        self._n = int(n)
        h, w = int(spatial[0]), int(spatial[1])
        self._shape = (self._n, h, w)
        self.repeat = max(1, int(repeat))
        self._features = np.memmap(features_path, dtype=np.float32, mode="r", shape=self._shape)
        self._labels = np.memmap(labels_path, dtype=np.int64, mode="r", shape=(self._n,))

    def __getitem__(self, i):
        idx = i % self._n
        feat = torch.from_numpy(np.array(self._features[idx], copy=True))
        lab = int(self._labels[idx])
        return {"input_features": feat, "labels": torch.tensor(lab, dtype=torch.long)}

    def __len__(self):
        return self._n * self.repeat


def _load_existing_disk(data_path: str, repeat: int):
    obj = torch.load(data_path, weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid synthetic_whisper manifest at {data_path!r}")
    fmt = obj.get("format")
    if fmt == CHUNKED_FORMAT:
        return SyntheticWhisperChunkedData(
            obj["chunks_dir"],
            obj["n"],
            int(obj.get("chunk_size", CHUNK_SIZE_DEFAULT)),
            repeat=repeat,
        )
    if fmt == SHARDED_FORMAT:
        return SyntheticWhisperShardedData(
            obj["shards_dir"],
            obj["n"],
            [int(x) for x in obj["shard_sizes"]],
            repeat=repeat,
        )
    if fmt == MEMMAP_FORMAT:
        memmap_dir = obj["memmap_dir"]
        h, w = int(obj["features_spatial_shape"][0]), int(obj["features_spatial_shape"][1])
        n = int(obj["n"])
        return SyntheticWhisperMemmapData(
            os.path.join(memmap_dir, obj["features_file"]),
            os.path.join(memmap_dir, obj["labels_file"]),
            n,
            (h, w),
            repeat=repeat,
        )
    if fmt == SINGLE_FILE_FORMAT:
        return SyntheticWhisperData(obj["samples"], repeat=repeat)
    raise ValueError(
        f"Unknown synthetic_whisper manifest format {fmt!r} at {data_path!r}"
    )


def load_data(conf: config.Config):
    sc = conf.data_configs.synthetic_whisper
    data_path = sc.data_path
    num_labels = getattr(sc, "num_labels", 10)
    force_regenerate = getattr(sc, "force_regenerate", 0)
    repeat = getattr(sc, "repeat", 1)
    n_unique = max(1, int(getattr(sc, "num_unique_samples", 7680)))

    data_type = effective_synthetic_whisper_data_type(sc)
    chunk_size_cfg = max(1, int(getattr(sc, "chunk_size", CHUNK_SIZE_DEFAULT)))
    num_shards_cfg = max(1, int(getattr(sc, "num_shards", 4)))

    if data_type == "memory":
        batch_sz = max(1, int(getattr(conf, "batch_size", 4)))
        print(
            "=============================================================\n"
            f"synthetic_whisper (memory): {batch_sz} unique samples (from --batch_size; num_unique_samples "
            f"ignored in memory_only), repeat={repeat} (len={batch_sz * max(1, int(repeat))}), no file cache\n"
            "============================================================="
        )
        samples = _sample_list(batch_sz, num_labels)
        return SyntheticWhisperData(samples, repeat=repeat)

    if os.path.exists(data_path) and not force_regenerate:
        print(
            "=============================================================\n"
            f"Loading Existing Data (expected {n_unique} samples = num_unique_samples; mismatch if cache is stale)\n"
            f"data_type={data_type}\n"
            "============================================================="
        )
        return _load_existing_disk(data_path, repeat)

    if force_regenerate and os.path.exists(data_path):
        print(
            "=============================================================\n"
            "Force Regenerate: Overwriting existing data\n"
            "============================================================="
        )
    else:
        print(
            "=============================================================\n"
            f"Generating New Data ({n_unique} samples, num_unique_samples)  data_type={data_type}\n"
            "============================================================="
        )

    if data_type == "chunks":
        generate_samples(n_unique, data_path, num_labels, chunk_size_cfg)
    elif data_type == "shard":
        generate_sharded(n_unique, data_path, num_labels, num_shards_cfg)
    elif data_type == "memmap":
        generate_memmap(n_unique, data_path, num_labels)
    elif data_type == "single_file":
        generate_single_file(n_unique, data_path, num_labels, chunk_size_cfg)
    else:
        raise ValueError(f"Unhandled data_type for disk: {data_type!r}")

    return _load_existing_disk(data_path, repeat)
