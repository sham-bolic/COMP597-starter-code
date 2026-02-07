import os
import torch
import torch.utils.data
import src.config as config
from transformers import WhisperFeatureExtractor

data_load_name = "synthetic_whisper"

N_SAMPLES = 5500
SAMPLE_RATE = 16000
VOCAB_SIZE = 51865
LABEL_LEN = 100

def generate_samples(n, data_path):
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    samples = []
    for i in range(n):
        wav = (torch.rand(SAMPLE_RATE) * 2 - 1).tolist()
        input_features = feature_extractor(
            wav,
            sampling_rate = SAMPLE_RATE,
            return_tensors = "pt"
        )["input_features"][0]
        labels = torch.randint(0, VOCAB_SIZE, (LABEL_LEN,))
        samples.append({
            "input_features": input_features,
            "labels": labels
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
    data_path = conf.data_configs.synthetic_whisper.data_path

    if os.path.exists(data_path):
        samples = torch.load(data_path)
    else:
        samples = generate_samples(N_SAMPLES, data_path)
    return SyntheticWhisperData(samples)