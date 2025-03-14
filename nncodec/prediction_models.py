import os
import abc
import random
import tensorflow as tf
import numpy as np
from keras import mixed_precision
from keras import utils

from .models import Symbol, SymbolFrequency, Dictionary
from .settings import TF_SEED

class base_prediction_model(abc.ABC):
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


class testing_prediction_model(base_prediction_model):
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


class tf_prediction_model(base_prediction_model):
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

    def __init__(self, dictionary, keras_model, model_weights_path=None):
        """
        Initialize tf_prediction_model.
        
        Args:
            dictionary (Dictionary): A Dictionary instance containing Symbol objects.
            keras_model (tf.keras.Model): A pre-configured Keras model (must be derived from TFKerasModelBase).
            model_weights_path (str, optional): Path to model weights.
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
        
        # Build a symbol lookup.
        self._sorted_symbols = sorted(dictionary.symbols, key=lambda s: s.data)
        self._token_map = {symbol.data: idx for idx, symbol in enumerate(self._sorted_symbols)}
        
        # Set the provided Keras model.
        self.model = keras_model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy'
        )
        
        # Build dummy states by calling the model's helper method.
        self.dummy_states = self.model.get_dummy_states()
        
        # Build the model by calling it with dummy inputs.
        main_input = tf.zeros((self.model.batch_size, self.model.seq_length), dtype=tf.int32)
        dummy_inputs = [main_input] + self.dummy_states
        _ = self.model(dummy_inputs)
        
        if model_weights_path is not None:
            self.load_model(model_weights_path)

    @tf.function(jit_compile=True)
    def _predict_raw(self, full_inputs):
        return self.model(full_inputs, training=False)

    def _symbols_to_tokens(self, symbols, dictionary):
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
        Convert a list of Symbol objects into a padded tensor using the model's
        seq_length and batch_size.
        """
        tokens = self._symbols_to_tokens(symbols, self.dictionary)
        sequences = [tokens]
        padded_sequences = utils.pad_sequences(
            sequences, maxlen=self.model.seq_length, padding='post', truncating='post', value=0
        )
        padded_sequences = tf.convert_to_tensor(padded_sequences, dtype=tf.int32)
        if padded_sequences.shape[0] != self.model.batch_size:
            padded_sequences = np.tile(padded_sequences, (self.model.batch_size, 1))
        return tf.convert_to_tensor(padded_sequences, dtype=tf.int32)

    def _postprocess_predictions(self, raw_output):
        if raw_output is None:
            raise ValueError("raw_output cannot be None")
        if isinstance(raw_output, tf.Tensor):
            raw_output = raw_output.numpy()
        if not isinstance(raw_output, np.ndarray):
            raise ValueError("raw_output must be a tf.Tensor or np.ndarray")
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
        if symbols is None or not isinstance(symbols, list):
            raise ValueError("symbols must be a list of Symbol objects")
        main_input_tensor = self._preprocess_input(symbols)
        full_inputs = [main_input_tensor] + self.dummy_states
        raw_output = self.model(full_inputs)
        predictions = self._postprocess_predictions(raw_output)
        return predictions[0]

    @tf.function
    def _train_step(self, full_inputs, target_tensor):
        with tf.GradientTape() as tape:
            predictions = self.model(full_inputs, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()(target_tensor, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, previous_symbols, correct_symbol):
        if previous_symbols is None or not isinstance(previous_symbols, list):
            raise ValueError("previous_symbols must be a list of Symbol objects")
        if not isinstance(correct_symbol, Symbol):
            raise ValueError("correct_symbol must be an instance of Symbol")
        main_input_tensor = self._preprocess_input(previous_symbols)
        full_inputs = [main_input_tensor] + self.dummy_states
        target_tokens = self._symbols_to_tokens([correct_symbol], self.dictionary)
        target_token = target_tokens[0]
        target_tensor = tf.fill([self.model.batch_size], target_token)
        loss = self._train_step(full_inputs, target_tensor)
        tf.print("Training loss:", loss)

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

