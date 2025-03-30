import os
import abc
import random
import tensorflow as tf
import numpy as np
from keras import mixed_precision
from keras import utils

from .models import Symbol, SymbolFrequency, Dictionary
from .settings import TF_SEED
from .logger import *

class BasePredictionModel(abc.ABC):
    def __init__(self, dictionary):
        """
        Initialize the prediction model.
        
        Args:
            dictionary (Dictionary): A Dictionary instance containing Symbol objects.
        """
        pass

    @abc.abstractmethod
    def predict(self, symbols, dictionary):
        pass

    @abc.abstractmethod
    def train(self, in_symbols, correct_symbol):
        pass

    @abc.abstractmethod
    def save_model(self, path):
        pass

    @abc.abstractmethod
    def load_model(self, path):
        pass
    
    @abc.abstractmethod
    def get_config_code(self):
        pass


class testing_prediction_model(BasePredictionModel):
    """A simple testing model (not using Keras) for unit tests."""
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def predict(self, context, dictionary=None):
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

    def train(self, in_symbols, correct_symbol):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass
    
    def get_config_code(self):
        return 0


class TfPredictionModel(BasePredictionModel):
    def enable_determinism(self, seed):
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

    def __init__(self, dictionary, keras_model, model_weights_path=None, use_weighted_average=False, logger=None, freeze_training=False):
        """
        Initialize TfPredictionModel.
        
        Args:
            dictionary (Dictionary): A Dictionary instance containing Symbol objects.
            keras_model (tf.keras.Model): A pre-configured Keras model.
            model_weights_path (str, optional): Path to model weights.
            use_weighted_average (bool, optional): Whether to combine predictions using a weighted average.
            freeze_training (bool, optional): If True, training updates are skipped.
        """
        if dictionary is None or not isinstance(dictionary, Dictionary):
            raise ValueError("dictionary must be a Dictionary instance and cannot be None")
        if keras_model is None or not isinstance(keras_model, tf.keras.Model):
            raise ValueError("keras_model must be a tf.keras.Model instance and cannot be None")
        
        # GPU configuration.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        print("GPUs Available:", tf.config.list_physical_devices('GPU'))
        
        self.enable_determinism(TF_SEED)
        self.dictionary = dictionary
        mixed_precision.set_global_policy('mixed_float16')
        
        self._sorted_symbols = sorted(dictionary.symbols, key=lambda s: s.data)
        self._token_map = {symbol.data: idx for idx, symbol in enumerate(self._sorted_symbols)}
        
        self.model = keras_model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy'
        )
        
        self.dummy_states = self.model.get_dummy_states()
        if self.dummy_states:
            self.stateful = True
            main_input = tf.zeros((self.model.batch_size, self.model.seq_length), dtype=tf.int32)
            dummy_inputs = [main_input] + self.dummy_states
            _ = self.model(dummy_inputs, training=False, return_states=True)
        else:
            self.stateful = False
            dummy_input = tf.zeros((self.model.batch_size, self.model.seq_length), dtype=tf.int32)
            _ = self.model(dummy_input, training=False)
        
        self.use_weighted_average = use_weighted_average
        self.freeze_training = freeze_training
        
        if model_weights_path is not None:
            self.load_model(model_weights_path)
            
        self.logger = logger

    @tf.function(jit_compile=True)
    def _predict_raw(self, full_inputs):
        if self.stateful:
            output = self.model(full_inputs, training=False, return_states=True)
            return output[0]
        else:
            return self.model(full_inputs, training=False)
    
    def _symbols_to_tokens(self, symbols, dictionary):
        """
        Convert a list of Symbol objects to tokens using zero-based indexing.
        """
        if not isinstance(symbols, list):
            raise ValueError("symbols must be a list of Symbol objects")
        if not all(isinstance(s, Symbol) for s in symbols):
            raise ValueError("All elements in symbols must be instances of Symbol")
        if not hasattr(dictionary, 'symbols'):
            raise ValueError("dictionary must have a 'symbols' attribute")
        tokens = []
        for symbol in symbols:
            token = self._token_map.get(symbol.data)
            if token is None:
                raise ValueError(f"Symbol with data {symbol.data} not found in dictionary.")
            tokens.append(token)
        return tokens

    def _preprocess_input(self, symbols):
        """
        Convert a list of Symbol objects into a padded tensor of shape (batch_size, seq_length).
        """
        tokens = self._symbols_to_tokens(symbols, self.dictionary)
        total_required = self.model.batch_size * self.model.seq_length
        if len(tokens) > total_required:
            tokens = tokens[-total_required:]
        elif len(tokens) < total_required:
            pad_length = total_required - len(tokens)
            tokens = [0] * pad_length + tokens  
        tokens_np = np.array(tokens).reshape((self.model.batch_size, self.model.seq_length))
        return tf.convert_to_tensor(tokens_np, dtype=tf.int32)

    def _postprocess_predictions(self, raw_output):
        """
        Convert raw model output into lists of SymbolFrequency objects.
        Tiles the output if its first dimension does not equal the model’s batch_size.
        """
        if raw_output is None:
            raise ValueError("raw_output cannot be None")
        if isinstance(raw_output, tf.Tensor):
            raw_output = raw_output.numpy()
        if not isinstance(raw_output, np.ndarray):
            raise ValueError("raw_output must be a tf.Tensor or np.ndarray")
        if raw_output.shape[0] != self.model.batch_size:
            raw_output = np.tile(raw_output, (self.model.batch_size, 1))
        batch_result = []
        for row in raw_output:
            symbol_freq_list = []
            for idx, prob in enumerate(row):
                try:
                    symbol = self._sorted_symbols[idx]
                except IndexError:
                    raise ValueError("Index out of range while mapping predictions to symbols.")
                symbol_freq_list.append(SymbolFrequency(symbol, prob))
            symbol_freq_list.sort(key=lambda sf: sf.frequency, reverse=True)
            batch_result.append(symbol_freq_list)
        return batch_result

    def predict(self, symbols):
        """
        Predict the next symbol(s) from a list of input symbols.
        If the underlying model is stateful, update internal LSTM states.
        """
        if symbols is None or not isinstance(symbols, list):
            raise ValueError("symbols must be a list of Symbol objects")
        main_input_tensor = self._preprocess_input(symbols)
        if self.stateful:
            full_inputs = [main_input_tensor] + self.dummy_states
            output = self.model(full_inputs, training=False, return_states=True)
            predictions = output[0]
            new_states = output[1:]
            self.dummy_states = new_states  # update state
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
    def _train_step(self, full_inputs, target_tensor):
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

    def train(self, previous_symbols, correct_symbol):
        """
        Train the model on the given symbol and update states if stateful.
        """
        if self.freeze_training:
            return
        if previous_symbols is None or not isinstance(previous_symbols, list):
            raise ValueError("previous_symbols must be a list of Symbol objects")
        if not isinstance(correct_symbol, Symbol):
            raise ValueError("correct_symbol must be an instance of Symbol")
        main_input_tensor = self._preprocess_input(previous_symbols)
        if self.stateful:
            full_inputs = [main_input_tensor] + self.dummy_states
        else:
            full_inputs = main_input_tensor
        target_tokens = self._symbols_to_tokens([correct_symbol], self.dictionary)
        target_token = target_tokens[0]
        target_tensor = tf.fill([self.model.batch_size], target_token)
        loss, new_states = self._train_step(full_inputs, target_tensor)
        if self.stateful and new_states is not None:
            self.dummy_states = new_states  # update state
        loss_value = loss.numpy() if hasattr(loss, 'numpy') else loss
        if self.logger is not None:
            self.logger.log(PredictionModelTrainingLog(loss_value))
        
    def get_config_code(self):
        return self.model.keras_code

    def save_model(self, path):
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

    def load_model(self, path):
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
