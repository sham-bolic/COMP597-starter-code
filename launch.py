import logging
import os
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelNamesMapping().get(os.environ.get("COMP597_LOG_LEVEL", "WARNING"), logging.WARNING),
    format="[{levelname:.4}] : {asctime} : {module:<24.24} : {message}",
    datefmt="%Y-%m-%dT%H:%M:%S",
    style='{',
)

from typing import Any, Dict, Optional, Tuple
import argparse
import gc
import src.config as config
import src.data as data
import src.models as models
import src.trainer as trainer
import src.trainer.stats as trainer_stats

def setup_logging(conf : config.Config) -> None:
    logging.basicConfig(
        filename=conf.logging.filename,
        filemode=conf.logging.filemode,
        format=conf.logging.format,
        datefmt=conf.logging.datefmt,
        style=conf.logging.style,
        level=conf.logging.level,
        force=True,
    )

def process_conf(conf : config.Config) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    dataset = data.load_data(conf)
    logger.debug(f"Dataset loaded with {len(dataset)} samples.")

    return models.model_factory(conf, dataset)

def get_conf() -> config.Config:
    parser = argparse.ArgumentParser()

    conf = config.Config()
    conf.add_arguments(parser)
    
    args, _ = parser.parse_known_args()
    conf.parse_arguments(args)
    return conf

def main():
    conf = get_conf()
    setup_logging(conf)
    logger.debug(f"Configuration: \n{conf}")
    logger.info(f"available models: {models.get_available_models()}")
    logger.info(f"available data load functions: {data.get_available_data_load_functions()}")
    logger.info(f"available trainer stats classes: {trainer_stats.get_available_trainer_stats()}")
    model_trainer, model_kwargs = process_conf(conf)
    model_trainer.train(model_kwargs)

    # This forces garbage collection at process exit. It ensure proper closing of resources.
    del conf
    del model_kwargs
    del model_trainer

if __name__ == "__main__":
    # Use 'spawn' to avoid "too many fds" when DataLoader uses num_workers > 0.
    # forkserver/fork inherit file descriptors; spawn starts fresh processes.
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # already set
    main()
    gc.collect()

