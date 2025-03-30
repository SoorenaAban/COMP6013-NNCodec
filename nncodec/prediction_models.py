"""
prediction_models.py

"""


import os
import abc
import random
import tensorflow as tf
import numpy as np
from typing import Any, List, Tuple, Optional, Union

from keras import mixed_precision

from .models import Symbol, SymbolFrequency, Dictionary
from .settings import SEED
from .logger import Logger, PredictionModelTrainingProgressStep, PredictionModelTrainingLog
from .keras_models import TFKerasModelBase
from .validators import validate_type


class BasePredictionModel(abc.ABC):
    """
    Abstract base class for prediction models.
    """

    def __init__(self, dictionary: Dictionary) -> None:
        """
        Initialize the prediction model.
        
        Args:
            dictionary (Dictionary): A Dictionary instance containing Symbol objects.
        """
        validate_type(dictionary, "dictionary", Dictionary)
        self.dictionary: Dictionary = dictionary

    @abc.abstractmethod
    def predict(self, symbols: List[Symbol]) -> Any:
        """
        Predict the next symbol(s) given the input symbols.
        
        Args:
            symbols (List[Symbol]): The context symbols.
        
        Returns:
            Any: Model-specific prediction result.
        """
        pass

    @abc.abstractmethod
    def train(self, in_symbols: List[Symbol], correct_symbol: Symbol) -> None:
        """
        Train the prediction model on the provided input and correct symbol.
        
        Args:
            in_symbols (List[Symbol]): The context symbols.
            correct_symbol (Symbol): The target symbol.
        """
        pass

    @abc.abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the model weights to the specified path.
        
        Args:
            path (str): The path to save the model weights.
        """
        pass

    @abc.abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load the model weights from the specified path.
        
        Args:
            path (str): The path from which to load the model weights.
        """
        pass
    
    @abc.abstractmethod
    def get_config_code(self) -> int:
        """
        Get a configuration code for the prediction model.
        
        Returns:
            int: The configuration code.
        """
        pass


class TestingPredictionModel(BasePredictionModel):
    """
    A simple testing prediction model (non-Keras based) for unit tests.
    """
    def __init__(self, dictionary: Dictionary) -> None:
        super().__init__(dictionary)
    
    def predict(self, context: List[Symbol], dictionary: Optional[Dictionary] = None) -> List[SymbolFrequency]:
        """
        Generate a dummy probability distribution based on the context length.
        
        Args:
            context (List[Symbol]): The input symbol context.
            dictionary (Optional[Dictionary]): Not used; defaults to self.dictionary.
        
        Returns:
            List[SymbolFrequency]: A list of SymbolFrequency objects sorted by frequency.
        """
        context_length = len(context)
        vocab_size = self.dictionary.get_size()
        delta = (context_length % 3) * 0.05
        if vocab_size == 3:
            probs = np.array([0.1 + delta, 0.3, 0.6 - delta], dtype=np.float64)
        else:
            base = np.linspace(1, vocab_size, num=vocab_size, dtype=np.float64)
            base[0] += delta
            probs = base / np.sum(base)
        probs = probs / np.sum(probs)
        sorted_symbols = sorted(self.dictionary.symbols, key=lambda s: s.data)
        if len(sorted_symbols) != len(probs):
            raise ValueError("Mismatch between number of symbols and probabilities.")
        result = [SymbolFrequency(sorted_symbols[i], float(probs[i])) for i in range(len(probs))]
        return result

    def train(self, in_symbols: List[Symbol], correct_symbol: Symbol) -> None:
        pass

    def save_model(self, path: str) -> None:
        pass

    def load_model(self, path: str) -> None:
        pass
    
    def get_config_code(self) -> int:
        return 0


class TfPredictionModel(BasePredictionModel):
    """
    Prediction model that utilizes a Keras model subclass (typically a subclass of TFKerasModelBase)
    to generate predictions and perform training.
    """
    
    def enable_determinism(self, seed: int) -> None:
        """
        Initialize random seeds for determinism.
        
        Args:
            seed (int): The seed value.
        """
        if seed is None:
            raise ValueError("seed cannot be None")
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()

    def __init__(
        self, 
        dictionary: Dictionary, 
        keras_model: TFKerasModelBase, 
        model_weights_path: Optional[str] = None, 
        use_weighted_average: bool = False, 
        logger: Optional[Logger] = None, 
        freeze_training: bool = False
    ) -> None:
        """
        Initialize the TensorFlow-based prediction model.
        
        Args:
            dictionary (Dictionary): A Dictionary instance containing Symbol objects.
            keras_model (TFKerasModelBase): A pre-configured Keras model (typically a subclass of TFKerasModelBase).
            model_weights_path (Optional[str]): Path to model weights.
            use_weighted_average (bool): Whether to combine predictions using a weighted average.
            logger (Optional[Logger]): Logger instance for logging.
            freeze_training (bool): If True, skip training updates.
        """
        validate_type(dictionary, "dictionary", Dictionary)
        validate_type(keras_model, "keras_model", tf.keras.Model)
        
        super().__init__(dictionary)
        # GPU configuration.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        print("GPUs Available:", tf.config.list_physical_devices('GPU'))
        
        self.enable_determinism(SEED)
        mixed_precision.set_global_policy('mixed_float16')
        
        self._sorted_symbols: List[Symbol] = sorted(dictionary.symbols, key=lambda s: s.data)
        self._token_map: dict = {symbol.data: idx for idx, symbol in enumerate(self._sorted_symbols)}
        
        self.model: TFKerasModelBase = keras_model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy'
        )
        
        self.dummy_states: List[tf.Tensor] = self.model.get_dummy_states()
        if self.dummy_states:
            self.stateful: bool = True
            main_input = tf.zeros((self.model.batch_size, self.model.seq_length), dtype=tf.int32)
            dummy_inputs = [main_input] + self.dummy_states
            _ = self.model(dummy_inputs, training=False, return_states=True)
        else:
            self.stateful = False
            dummy_input = tf.zeros((self.model.batch_size, self.model.seq_length), dtype=tf.int32)
            _ = self.model(dummy_input, training=False)
        
        self.use_weighted_average: bool = use_weighted_average
        self.freeze_training: bool = freeze_training
        
        if model_weights_path is not None:
            self.load_model(model_weights_path)
            
        self.logger: Optional[Logger] = logger

    @tf.function(jit_compile=True)
    def _predict_raw(self, full_inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        """
        Run a raw prediction step.
        
        Args:
            full_inputs (tf.Tensor or List[tf.Tensor]): The model input(s).
        
        Returns:
            tf.Tensor: The raw output tensor from the model.
        """
        if self.stateful:
            output = self.model(full_inputs, training=False, return_states=True)
            return output[0]
        else:
            return self.model(full_inputs, training=False)
    
    def _symbols_to_tokens(self, symbols: List[Symbol], dictionary: Dictionary) -> List[int]:
        """
        Convert a list of Symbol objects to tokens using zero-based indexing.
        
        Args:
            symbols (List[Symbol]): The list of symbols.
            dictionary (Dictionary): The dictionary to use for mapping.
        
        Returns:
            List[int]: The corresponding token indices.
        
        Raises:
            ValueError: If any symbol is not found in the dictionary.
        """
        validate_type(symbols, "symbols", list)
        if not all(isinstance(s, Symbol) for s in symbols):
            raise ValueError("All elements in symbols must be instances of Symbol")
        if not hasattr(dictionary, 'symbols'):
            raise ValueError("dictionary must have a 'symbols' attribute")
        tokens: List[int] = []
        for symbol in symbols:
            token = self._token_map.get(symbol.data)
            if token is None:
                raise ValueError(f"Symbol with data {symbol.data} not found in dictionary.")
            tokens.append(token)
        return tokens

    def _preprocess_input(self, symbols: List[Symbol]) -> tf.Tensor:
        """
        Convert a list of Symbol objects into a padded tensor of shape (batch_size, seq_length).
        
        Args:
            symbols (List[Symbol]): The input symbols.
        
        Returns:
            tf.Tensor: The preprocessed input tensor.
        """
        tokens: List[int] = self._symbols_to_tokens(symbols, self.dictionary)
        total_required = self.model.batch_size * self.model.seq_length
        if len(tokens) > total_required:
            tokens = tokens[-total_required:]
        elif len(tokens) < total_required:
            pad_length = total_required - len(tokens)
            tokens = [0] * pad_length + tokens  
        tokens_np = np.array(tokens).reshape((self.model.batch_size, self.model.seq_length))
        return tf.convert_to_tensor(tokens_np, dtype=tf.int32)

    def _postprocess_predictions(self, raw_output: Union[tf.Tensor, np.ndarray]) -> List[List[SymbolFrequency]]:
        """
        Convert raw model output into lists of SymbolFrequency objects.
        Tiles the output if its first dimension does not equal the model's batch_size.
        
        Args:
            raw_output (tf.Tensor or np.ndarray): The raw predictions.
        
        Returns:
            List[List[SymbolFrequency]]: A batch of symbol frequency lists.
        """
        if raw_output is None:
            raise ValueError("raw_output cannot be None")
        if isinstance(raw_output, tf.Tensor):
            raw_output = raw_output.numpy()
        if not isinstance(raw_output, np.ndarray):
            raise ValueError("raw_output must be a tf.Tensor or np.ndarray")
        if raw_output.shape[0] != self.model.batch_size:
            raw_output = np.tile(raw_output, (self.model.batch_size, 1))
        batch_result: List[List[SymbolFrequency]] = []
        for row in raw_output:
            symbol_freq_list: List[SymbolFrequency] = []
            for idx, prob in enumerate(row):
                try:
                    symbol = self._sorted_symbols[idx]
                except IndexError:
                    raise ValueError("Index out of range while mapping predictions to symbols.")
                symbol_freq_list.append(SymbolFrequency(symbol, prob))
            symbol_freq_list.sort(key=lambda sf: sf.frequency, reverse=True)
            batch_result.append(symbol_freq_list)
        return batch_result

    def predict(self, symbols: List[Symbol]) -> List[SymbolFrequency]:
        """
        Predict the next symbol(s) from a list of input symbols.
        If the underlying model is stateful, update internal states.
        
        Args:
            symbols (List[Symbol]): The context symbols.
        
        Returns:
            List[SymbolFrequency]: A list of symbol frequencies for the predicted output.
        """
        validate_type(symbols, "symbols", list)
        main_input_tensor: tf.Tensor = self._preprocess_input(symbols)
        if self.stateful:
            full_inputs: List[tf.Tensor] = [main_input_tensor] + self.dummy_states
            output = self.model(full_inputs, training=False, return_states=True)
            predictions = output[0]
            new_states = output[1:]
            self.dummy_states = new_states 
        else:
            full_inputs = main_input_tensor
            predictions = self.model(full_inputs, training=False)
        raw_output_np = predictions.numpy() if isinstance(predictions, tf.Tensor) else predictions
        if not self.use_weighted_average:
            selected_output = raw_output_np[-1]
        else:
            weights = np.arange(1, raw_output_np.shape[0] + 1, dtype=np.float32)
            weights /= np.sum(weights)
            selected_output = np.average(raw_output_np, axis=0, weights=weights)
        selected_output = np.expand_dims(selected_output, axis=0)
        predictions_post = self._postprocess_predictions(selected_output)
        if self.logger is not None:
            self.logger.log(PredictionModelTrainingProgressStep(len(symbols)))
        return predictions_post[0]
    
    @tf.function
    def _train_step(self, full_inputs: Union[tf.Tensor, List[tf.Tensor]], target_tensor: tf.Tensor) -> Tuple[tf.Tensor, Optional[List[tf.Tensor]]]:
        """
        Perform a single training step.
        
        Args:
            full_inputs (tf.Tensor or List[tf.Tensor]): The input(s) to the model.
            target_tensor (tf.Tensor): The target tensor.
        
        Returns:
            Tuple[tf.Tensor, Optional[List[tf.Tensor]]]: The loss and, if stateful, the new states.
        """
        if self.stateful:
            with tf.GradientTape() as tape:
                output = self.model(full_inputs, training=True, return_states=True)
                predictions = output[0]
                new_states = output[1:]
                loss = tf.keras.losses.SparseCategoricalCrossentropy()(target_tensor, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss, new_states
        else:
            with tf.GradientTape() as tape:
                predictions = self.model(full_inputs, training=True)
                loss = tf.keras.losses.SparseCategoricalCrossentropy()(target_tensor, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss, None

    def train(self, previous_symbols: List[Symbol], correct_symbol: Symbol) -> None:
        """
        Train the model on the given symbol and update states if stateful.
        
        Args:
            previous_symbols (List[Symbol]): The context symbols.
            correct_symbol (Symbol): The target symbol.
        """
        if self.freeze_training:
            return
        validate_type(previous_symbols, "previous_symbols", list)
        validate_type(correct_symbol, "correct_symbol", Symbol)
        main_input_tensor: tf.Tensor = self._preprocess_input(previous_symbols)
        if self.stateful:
            full_inputs: List[tf.Tensor] = [main_input_tensor] + self.dummy_states
        else:
            full_inputs = main_input_tensor
        target_tokens: List[int] = self._symbols_to_tokens([correct_symbol], self.dictionary)
        target_token: int = target_tokens[0]
        target_tensor: tf.Tensor = tf.fill([self.model.batch_size], target_token)
        loss, new_states = self._train_step(full_inputs, target_tensor)
        if self.stateful and new_states is not None:
            self.dummy_states = new_states
        loss_value = loss.numpy() if hasattr(loss, 'numpy') else loss
        if self.logger is not None:
            self.logger.log(PredictionModelTrainingLog(loss_value))
        
    def get_config_code(self) -> int:
        return self.model.keras_code

    def save_model(self, path: str) -> None:
        """
        Save the model weights to the specified path.
        
        Args:
            path (str): The path to save the weights.
        """
        if not isinstance(path, str) or len(path.strip()) == 0:
            raise ValueError("The provided path must be a non-empty string.")
        if not path.endswith(".weights.h5"):
            path = path + ".weights.h5"
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                raise ValueError(f"Failed to create directory {directory}: {e}")
        if directory and not os.access(directory, os.W_OK):
            raise ValueError(f"The directory {directory} is not writable.")
        self.model.save_weights(path)

    def load_model(self, path: str) -> None:
        """
        Load model weights from the specified path.
        
        Args:
            path (str): The path from which to load the weights.
        """
        if not isinstance(path, str) or len(path.strip()) == 0:
            raise ValueError("The provided path must be a non-empty string.")
        if not path.endswith(".weights.h5"):
            path = path + ".weights.h5"
        if not os.path.exists(path):
            raise ValueError(f"The file {path} does not exist.")
        try:
            self.model.load_weights(path)
        except Exception as e:
            raise ValueError(f"Failed to load weights from {path}: {e}")
