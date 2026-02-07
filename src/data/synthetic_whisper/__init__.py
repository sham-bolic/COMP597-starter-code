"""Thin wrapper around the datasets module.

This module is a very thin wrapper around the Hugging Face datasets module. It 
makes the translation between the config object and the datasets module, and it 
will hopefully make it easy to extend in the future if needs be.

"""
from src.data.synthetic_whisper.data import *
