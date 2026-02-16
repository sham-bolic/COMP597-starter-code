# === import necessary modules ===
import src.config as config # Configurations
import src.trainer as trainer # Trainer base class
import src.trainer.stats as trainer_stats # Trainer statistics module

# === import necessary external modules ===
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers

from transformers import AutoProcessor, WhisperConfig, WhisperForAudioClassification

"""
This file contains the code to train a whisper-tiny model for audio classification
using Simple trainer (src/trainer/simple.py).
It is based on the whisper-tiny model from HuggingFace Transformers.
https://huggingface.co/openai/whisper-tiny
"""

def whisper_collator(batch):
    input_features = torch.stack([item["input_features"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "input_features": input_features,
        "labels": labels,
    }

def whisper_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    num_labels = getattr(
        conf.data_configs.synthetic_whisper, "num_labels", 10
    )

    model_config = WhisperConfig.from_pretrained("openai/whisper-tiny")
    model_config.num_labels = num_labels

    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForAudioClassification.from_pretrained(
        "openai/whisper-tiny", config=model_config
    )

    loader = data.DataLoader(
        dataset,
        batch_size = conf.batch_size,
        collate_fn = whisper_collator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(),  lr = conf.learning_rate)

    scheduler = transformers.get_scheduler(
        "linear",
        optimizer = optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader), 
    )

    return trainer.SimpleTrainer(
        loader = loader,
        model = model,
        optimizer = optimizer,
        lr_scheduler = scheduler,
        device = device,
        stats = trainer_stats.init_from_conf(
            conf = conf,
            device = device,
            num_train_steps = len(loader)
            )
    ), None