import abc
from typing import List, Optional, Any

from .prediction_models import TfPredictionModel
from .models import Symbol
from .logger import Logger


class BaseTrainer(abc.ABC):
    """
    Abstract base class for trainers.
    """

    @abc.abstractmethod
    def train(self, 
              prediction_model: TfPredictionModel, 
              symbols_collection: List[Symbol], 
              logger: Optional[Logger] = None) -> None:
        """
        Train the prediction model on a collection of symbols.
        
        For each symbol in the collection, the model is trained using the previous symbols as input.
        For the first symbol, an empty list is used.
        
        Args:
            prediction_model (TfPredictionModel): The prediction model to train.
            symbols_collection (List[Symbol]): A list of symbols used for training.
            logger (Optional[Logger]): An optional logger for progress reporting.
        """
        pass


class TfTrainer(BaseTrainer):
    """
    Trainer for TensorFlow-based prediction models.
    """

    def train(self, 
              prediction_model: TfPredictionModel, 
              symbols_collection: List[Symbol], 
              logger: Optional[Logger] = None) -> None:
        if not isinstance(symbols_collection, list):
            raise ValueError("symbols_collection must be a list of symbols.")
        if len(symbols_collection) == 0:
            raise ValueError("symbols_collection cannot be empty.")
        for i, symbol in enumerate(symbols_collection):
            previous_symbols = symbols_collection[:i]
            prediction_model.train(previous_symbols, symbol)
