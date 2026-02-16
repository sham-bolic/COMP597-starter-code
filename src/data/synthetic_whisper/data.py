import os
import torch
import torch.utils.data
import src.config as config
from transformers import WhisperFeatureExtractor

data_load_name = "synthetic_whisper"

N_SAMPLES = 5500
SAMPLE_RATE = 16000

def generate_samples(n, data_path, num_labels):
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    samples = []
    for i in range(n):
        wav = (torch.rand(SAMPLE_RATE) * 2 - 1).tolist()
        input_features = feature_extractor(
            wav,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )["input_features"][0]
        label = torch.randint(0, num_labels, ())
        samples.append({
            "input_features": input_features,
            "labels": label
        })

    torch.save(samples, data_path)
    return samples

class SyntheticWhisperData(torch.utils.data.Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)

def load_data(conf: config.Config):
    sc = conf.data_configs.synthetic_whisper
    data_path = sc.data_path
    num_labels = getattr(sc, "num_labels", 10)
    force_regenerate = getattr(sc, "force_regenerate", 0)

    if os.path.exists(data_path) and not force_regenerate:
        print(f'=============================================================\nLoading Existing Data\n=============================================================')
        samples = torch.load(data_path)
    else:
        if force_regenerate and os.path.exists(data_path):
            print(f'=============================================================\nForce Regenerate: Overwriting existing data\n=============================================================')
        else:
            print(f'=============================================================\nGenerating New Data\n=============================================================')
        samples = generate_samples(N_SAMPLES, data_path, num_labels)
    return SyntheticWhisperData(samples)