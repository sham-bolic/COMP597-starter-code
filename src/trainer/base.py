from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import src.trainer.stats as trainer_stats
import torch
import torch.nn as nn
import torch.utils.data as data
import tqdm.auto

class Trainer(ABC):
    """Base class implemented by all trainer objects.

    This abstract class defines the methods and basic attributes of a trainer 
    class.

    Parameters
    ----------
    model
        The model to train.
    loader
        A PyTorch dataloader that will be used to obtain the data at each step.
    device
        The device on which the model resides and where batches will be moved to.
    stats
        An object to gather statistics during training. If ``None``, a
        ``NOOPTrainerStats`` instance is created for this ``device`` (writes
        ``baseline_train_duration*.txt`` when the train loop finishes; see
        ``src/trainer/stats/noop.py``).
    enable_checkpointing
        Whether or not to checkpoint the model.
    checkpoint_frequency
        The after how many steps a checkpoint is saved.

    Attributes
    ----------
    model : torch.nn.Module
        The model to train.
    loader : torch.utils.data.DataLoader
        The object used to load data during training.
    device : torch.device
        The device on which the model resides and where batches will be moved to.
    stats : src.trainer.stats.TrainerStats
        The `TrainerStats` object used to gather statistics.
    enable_checkpointing : bool
        Whether or not the model will be checkpointed during training.
    checkpoint_frequency : int
        After how many steps a checkpoint is saved.

    """

    def __init__(self, 
                 model : nn.Module, 
                 loader : data.DataLoader, 
                 device : torch.device, 
                 stats : Optional[trainer_stats.TrainerStats] = None, 
                 enable_checkpointing : bool = False,
                 checkpoint_frequency : int = 1):
        self.model = model
        self.loader = loader
        self.device = device
        if stats is None:
            stats = trainer_stats.NOOPTrainerStats(
                device=device,
                output_path=Path("baseline_train_duration.txt"),
            )
        self.stats = stats
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_frequency = checkpoint_frequency

    def should_save_checkpoint(self, i : int) -> bool:
        """Condition to device when to save a checkpoint.

        Implements a condition to check if it is time to save a checkpoint 
        based on the iteration number. This allows customizing the behaviour 
        of when to save a checkpoint.

        Parameters
        ----------
        i
            This is the iteration number.

        """
        return (i + 1) % self.checkpoint_frequency == 0

    def checkpoint_path(self, i : int) -> str:
        """Generates the checkpoint path.

        This generates the path and filename of where to save the checkpoint. 
        This allows easy customization of how to name checkpoints and where to 
        save them.

        Parameters
        ----------
        i
            This is the iteration number.

        """
        return "checkpoint.tar"

    # TODO(Olivier) Consider adding the loss to the checkpoint.
    def checkpoint_dict(self, i : int) -> Dict[str,Any]:
        """Generate the dictionary of contents to save in the checkpoint.
        
        This generates a dictionary containing all the information that should 
        be saved in order to be able to restore from the checkpoint.

        Parameters
        ----------
        i
            This is the iteration number.

        Returns
        -------
        dict
            The dictionary that will be saved.
        """
        return {
            "step": i,
            "model_state_dict": self.model.state_dict(),
        }

    def save_checkpoint(self, i : int) -> None:
        """Save a checkpoint of the current training task.

        This is designed to save a snapshot of the current training task. It 
        should save the model state, optimizer state along with anything else 
        required to restart training from the checkpoint.

        Parameters
        ----------
        i 
            This is the iteration number.

        """
        path = self.checkpoint_path(i)
        checkpoint_dict = self.checkpoint_dict(i)
        torch.save(checkpoint_dict, path)
    
    def process_batch(self, i : int, batch : Any) -> Any:
        return {k: v.to(self.device) for k, v in batch.items()}
    
    @abstractmethod
    def forward(self, i : int, batch : Any, model_kwargs : Dict[str, Any]) -> torch.Tensor:
        """Execution of the model's forward pass.

        Parameters
        ----------
        i
            This is the iteration number.
        batch
            The data of the current batch. The type is defined by the 
            `DataLoader` provided when the object was constructed. It must be a 
            dictionary or similar as it is unpacked to the model using 
            `**batch`.
        model_kwargs
            Additional arguments that need to be provided to the model during 
            the forward pass.
        
        Returns
        -------
        torch.Tensor
            The loss computed during the iteration.
        
        """
        pass

    @abstractmethod
    def backward(self, i : int, loss : torch.Tensor) -> None:
        """Execution of the model's backward pass.

        Parameters
        ----------
        i
            The iteration number.
        loss
            The loss computed during the forward pass.
        
        """
        pass

    @abstractmethod
    def optimizer_step(self, i : int) -> None:
        """Execution of the optimizer step.

        Parameters
        ----------
        i
            The iteration number.
        """
        pass

    def step(self, i : int, batch : Any, model_kwargs : Optional[Dict[str, Any]]) -> Tuple[torch.Tensor, Optional[str]]:
        """Execution of a single training step.

        Parameters
        ----------
        i
            This is the iteration number.
        batch
            The data of the current batch. The type is defined by the 
            `DataLoader` provided when the object was constructed. It must be a 
            dictionary or similar as it is unpacked to the model using 
            `**batch`.
        model_kwargs
            Additional arguments that need to be provided to the model during 
            the forward pass.

        Returns
        -------
        torch.Tensor
            The loss computed during the iteration.
        str, optional
            Any additional information to print before updating the progress 
            bar. This allows allows properly printing an update without 
            breaking, overwriting or duplicating the progress bar.

        """
        if model_kwargs is None:
            model_kwargs = {}
        batch = self.process_batch(i, batch)

        self.stats.start_forward()
        loss = self.forward(i, batch, model_kwargs)
        self.stats.stop_forward()

        self.stats.start_backward()
        self.backward(i, loss)
        self.stats.stop_backward()

        self.stats.start_optimizer_step()
        self.optimizer_step(i)
        self.stats.stop_optimizer_step()
        
        return loss, None

    def train(self, model_kwargs : Optional[Dict[str, Any]]) -> None:
        """Training loop for the model.
        
        This will execute a training step on each batch provided by the 
        dataloader. The number of iterations is defined by the dataloader 
        provided when the object was constructed. 

        A progress bar is updated after every iteration with the iteration 
        number and the most recent loss.

        Training statistics will be logged after every iteration if the `stats` 
        attribute implements the method `log_step`. Additionally, more 
        statistics will be displayed at the end of training if the `stats` 
        attribute implements the method `log_stats`.

        Parameters
        ----------
        model_kwargs
            Additional arguments that need to be provided to the model during 
            the forward pass.

        Notes
        -----
            This does not support multi-epoch training. If you need training on 
            multiple epochs, you should implement a class that inherits 
            `Trainer` and overrides the `train` method.

        """
        progress_bar = tqdm.auto.tqdm(range(len(self.loader)), desc="loss: N/A")

        self.stats.start_train()
        for i, batch in enumerate(self.loader):
            self.stats.start_step()
            loss, descr = self.step(i, batch, model_kwargs)
            self.stats.stop_step()

            if self.enable_checkpointing and self.should_save_checkpoint(i):
                self.stats.start_save_checkpoint()
                self.save_checkpoint(i)
                self.stats.stop_save_checkpoint()

            # for every rank, log the loss
            self.stats.log_loss(loss)
            self.stats.log_step()

            if descr is not None:
                progress_bar.clear()
                print(descr)
            progress_bar.clear()
            progress_bar.update(1)

        self.stats.stop_train()
        progress_bar.close()
        self.stats.log_stats()
