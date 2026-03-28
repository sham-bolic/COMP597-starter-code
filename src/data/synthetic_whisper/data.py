import os
import torch
import torch.utils.data
import src.config as config
from transformers import WhisperFeatureExtractor

data_load_name = "synthetic_whisper"

N_SAMPLES = 500
SAMPLE_RATE = 16000


def _sample_list(n: int, num_labels: int, sample_rate: int = SAMPLE_RATE) -> list:
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    samples = []
    for _ in range(n):
        wav = (torch.rand(sample_rate) * 2 - 1).tolist()
        input_features = feature_extractor(
            wav,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )["input_features"][0]
        label = torch.randint(0, num_labels, ())
        samples.append({
            "input_features": input_features,
            "labels": label,
        })
    return samples


def generate_samples(n, data_path, num_labels):
    samples = _sample_list(n, num_labels)
    torch.save(samples, data_path)
    return samples

class SyntheticWhisperData(torch.utils.data.Dataset):

    def __init__(self, samples, repeat: int = 1):
        self.samples = samples
        self._n = len(samples)
        self.repeat = max(1, int(repeat))

    def __getitem__(self, i):
        return self.samples[i % self._n]

    def __len__(self):
        return self._n * self.repeat


def load_data(conf: config.Config):
    sc = conf.data_configs.synthetic_whisper
    data_path = sc.data_path
    num_labels = getattr(sc, "num_labels", 10)
    force_regenerate = getattr(sc, "force_regenerate", 0)
    memory_only = getattr(sc, "memory_only", 1)
    n_samples = max(1, int(getattr(sc, "n_samples", N_SAMPLES)))
    repeat = getattr(sc, "repeat", 1)

    if memory_only:
        print(
            "=============================================================\n"
            f"synthetic_whisper (memory_only): {n_samples} samples in memory, repeat={repeat} "
            f"(len={n_samples * max(1, int(repeat))}), no file cache\n"
            "============================================================="
        )
        samples = _sample_list(n_samples, num_labels)
        return SyntheticWhisperData(samples, repeat=repeat)

    if os.path.exists(data_path) and not force_regenerate:
        print(f'=============================================================\nLoading Existing Data\n=============================================================')
        samples = torch.load(data_path)
    else:
        if force_regenerate and os.path.exists(data_path):
            print(f'=============================================================\nForce Regenerate: Overwriting existing data\n=============================================================')
        else:
            print(f'=============================================================\nGenerating New Data\n=============================================================')
        samples = generate_samples(n_samples, data_path, num_labels)
    return SyntheticWhisperData(samples, repeat=repeat)