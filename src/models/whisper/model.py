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

from transformers import AutoProcessor, WhisperForConditionalGeneration

"""
This file contains the code to train a whisper-tiny model using Simple trainer (src/trainer/simple.py).
It is based on the whisper-tiny model from HuggingFace Transformers.
https://huggingface.co/openai/whisper-tiny
"""

def whisper_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    data_collator = transformers.DataCollatorForSpeechSeq2SeqWithPadding(
        processor = processor,
        decoder_start_token_id = model.config.decoder_start_token_id,
        forward_attention_mask = True
    )

    loader = data.DataLoader(
        dataset,
        batch_size = conf.batch_size,
        collate_fn = data_collator
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