"""Statistical tools to use along trainers.

Provides classes to accumulate data from trainers and provide basic analysis on 
the data.

"""
from typing import List
from src.trainer.stats.base import TrainerStats
from src.trainer.stats.noop import NOOPTrainerStats
from src.trainer.stats.simple import SimpleTrainerStats
from src.trainer.stats.codecarbon import CodeCarbonStats
from src.trainer.stats.resource_util import ResourceUtilStats
from src.trainer.stats.utils import *
import src.auto_discovery as auto_discovery
import src.config as config

_CONSTRUCTOR_FUNCTION_NAME="construct_trainer_stats"
_TRAINER_STATS_REGISTRATION_NAME="trainer_stats_name"
_TRAINER_STATS_IGNORE="_TRAINER_STATS_AUTO_DISCOVERY_IGNORE"
_TRAINER_STATS_CONSTRUCTORS=auto_discovery.register(
    package=__package__,
    path=list(__path__),
    module_attr_name=_CONSTRUCTOR_FUNCTION_NAME,
    name_override_attr_name=_TRAINER_STATS_REGISTRATION_NAME,
    ignore_attr_name=_TRAINER_STATS_IGNORE,
    strict_ispkg=False
)

def init_from_conf(conf : config.Config, **kwargs) -> TrainerStats:
    """Factory for initialize a `TrainerStats`.

    This is a factory that initializes a `TrainerStats` objects using a 
    configuration object. 

    Parameters
    ----------
    conf
        A configuration object.
    **kwargs
        This is used for when additional configurations are need for the 
        specified trainer stats type. Please see the implemented trainer stats 
        to know what extra args they might need.

    """
    global _TRAINER_STATS_CONSTRUCTORS
    constructor_fn = _TRAINER_STATS_CONSTRUCTORS.get(conf.trainer_stats, None)
    if constructor_fn is None:
        raise Exception(f"Unknown trainer stats format: {conf.trainer_stats}")
    return constructor_fn(conf, **kwargs)

def get_available_trainer_stats() -> List[str]:
    global _TRAINER_STATS_CONSTRUCTORS
    return [name for name in _TRAINER_STATS_CONSTRUCTORS.keys()]
