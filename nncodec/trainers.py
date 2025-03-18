# trainers.py

import abc
from .prediction_models import TfPredictionModel
from .models import Dictionary
from .keras_models import LstmKerasModel


class base_trainer(abc.ABC):
    @abc.abstractmethod
    def __init__(self, dictionary, model_config, model_weights_path=None):
        """
        Initializes the trainer with a symbols dictionary, prediction model configurations,
        and optionally the address of an existing model to train upon.
        
        Args:
            dictionary: The symbols dictionary (e.g. a models.Dictionary instance).
            model_config (dict): Configuration parameters for the prediction model.
            model_weights_path (str, optional): File path of an existing model to train upon.
        """
        pass

    @abc.abstractmethod
    def save(self, filepath):
        """
        Saves the model's weights to the given file path.
        
        Args:
            filepath (str): The file path to save the model weights.
        """
        pass

    @abc.abstractmethod
    def train(self, symbols_collection):
        """
        Trains the model on a collection of symbols.
        
        For each symbol in the collection, the model is trained using the previous symbols as input.
        For the first symbol, an empty list is used.
        
        Args:
            symbols_collection (list): A collection (list) of symbols to be used for training.
        """
        pass


class tf_trainer(base_trainer):
    def __init__(self, dictionary, keras_model, model_weights_path=None):
        self.prediction_model = TfPredictionModel(dictionary,
                                                    keras_model,
                                                    model_weights_path=model_weights_path)

    def save(self, filepath):
        if not isinstance(filepath, str) or not filepath.strip():
            raise ValueError("The provided filepath must be a non-empty string.")
        self.prediction_model.save_model(filepath)

    def train(self, symbols_collection):
        if not isinstance(symbols_collection, list):
            raise ValueError("symbols_collection must be a list of symbols.")
        if len(symbols_collection) == 0:
            raise ValueError("symbols_collection cannot be empty.")
        for i, symbol in enumerate(symbols_collection):
            previous_symbols = symbols_collection[:i]
            self.prediction_model.train(previous_symbols, symbol)
