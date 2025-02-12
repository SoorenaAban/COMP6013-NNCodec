#prediction-model.py

import os
import abc
import random
import tensorflow as tf
import numpy as np
from keras import mixed_precision
from keras import utils

from .models import Symbol, SymbolFrequency, Dictionary
from .settings import *

class base_prediction_model(abc.ABC):
    def __init__(self, dictionary, settings_override=None):
        """
        Initialize the TensorFlow prediction model.
        
        Args:
            dictionary (Dictionary): A Dictionary instance containing Symbol objects.
            settings_override (dict, optional): A dictionary with settings values to override the defaults.)
            (intended for testing)
                Expected keys: 
                    - 'TF_BATCH_SIZES'
                    - 'TF_SEQ_LENGTH'
                    - 'TF_NUM_LAYERS'
                    - 'TF_RNN_UNITS'
                    - 'TF_EMBEDING_SIZE'
                    - 'TF_START_LEARNING_RATE'
                If not provided, default settings are loaded from the global 'settings' module.
        """
        pass

    @abc.abstractmethod
    def predict(self, symbols, dictionary):
        """
        returns the predicted data
        
        Args:
            symbols(list[Symbol]): list of symbols to be used for prediction
            dictionary(Dictionary): the dictionary to be used for prediction

        Returns:
            np.ndarray: returns the predicted data
        """
        pass

    @abc.abstractmethod
    def train(self, in_symbols, correct_symbol):
        """
        trains the model
        
        Args:
            in_symbols(list[Symbol]): list of symbols to be used for training as input for prediction
            correct_symbol(Symbol): the correct symbol that should be predicted
        """
        pass

    @abc.abstractmethod
    def save_model(self, path):
        """
        saves the model to the given path
        
        Args:
            path(string): the path where the model should be saved
        """
        pass

    @abc.abstractmethod
    def load_model(self, path):
        """
        loads the model from the given path
        
        Args:
            path(string): the path from where the model should be loaded
        """
        pass

class testing_prediction_model(base_prediction_model):
    """This model is only intended for unit testing"""
    def __init__(self, dictionary, settings_override=None):
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
        return probs

    def train(self, in_symbols, correct_symbol):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass

class tf_prediction_model(base_prediction_model):
    
    def reset_seed(self, seed):
        """
        Initializes various random seeds to help with determinism.
        
        Args:
            seed(int): The seed value to use for random number generation.
            
        Raises:
            ValueError: If seed is None.
        """
        
        if(seed is None):
            raise ValueError("seed cannot be None")
        
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def __init__(self, dictionary, settings_override=None):
        if((dictionary is None) or (not isinstance(dictionary, Dictionary))):
            raise ValueError("dictionary must be a Dictionary instance and cannot be None")
        if(settings_override is not None and not isinstance(settings_override, dict)):
            raise ValueError("settings_override must be a dictionary if provided")
        
        # Set up GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        print("GPUs Available:", tf.config.list_physical_devices('GPU'))
        
        if settings_override is None:
            self._settings = {
                'TF_SEED': TF_SEED,
                'TF_BATCH_SIZES': TF_BATCH_SIZES,
                'TF_SEQ_LENGTH': TF_SEQ_LENGTH,
                'TF_NUM_LAYERS': TF_NUM_LAYERS,
                'TF_RNN_UNITS': TF_RNN_UNITS,
                'TF_EMBEDING_SIZE': TF_EMBEDING_SIZE,
                'TF_START_LEARNING_RATE': TF_START_LEARNING_RATE,
            }
        else:
            self._settings = settings_override

        self.reset_seed(self._settings['TF_SEED'])
        self.batch_size = self._settings['TF_BATCH_SIZES']
        self.seq_length = self._settings['TF_SEQ_LENGTH']
        self.num_layers = self._settings['TF_NUM_LAYERS']
        self.rnn_units = self._settings['TF_RNN_UNITS']
        self.embedding_size = self._settings['TF_EMBEDING_SIZE']
        self.start_learning_rate = self._settings['TF_START_LEARNING_RATE']

        self.dictionary = dictionary

        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
                
        self.dummy_states = []
        for _ in range(self.num_layers):
            dummy_state = tf.zeros((self.batch_size, self.rnn_units), dtype=tf.float16)
            self.dummy_states.extend([dummy_state, dummy_state])
        

        inputs = [tf.keras.Input(batch_shape=[self.batch_size, self.seq_length], name="main_input")]
        for i in range(self.num_layers):
            state_h = tf.keras.Input(shape=(self.rnn_units,), dtype=tf.float16, name=f'state_h_{i}')
            state_c = tf.keras.Input(shape=(self.rnn_units,), dtype=tf.float16, name=f'state_c_{i}')
            inputs.extend([state_h, state_c])

        embedded = tf.keras.layers.Embedding(
            self.dictionary.get_size(), 
            self.embedding_size,
            name="embedding"
        )(inputs[0])

        predictions, state_h, state_c = tf.keras.layers.LSTM(
            self.rnn_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
            name="lstm_0"
        )(embedded,initial_state=[inputs[1], inputs[2]])
        skip_connections = [predictions]
        outputs = [state_h, state_c]  

        for i in range(1, self.num_layers):
            layer_input = tf.keras.layers.concatenate(
                [embedded, skip_connections[-1]],
                name=f"concat_{i}"
            )
            predictions, state_h, state_c = tf.keras.layers.LSTM(
                self.rnn_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform',
                name=f"lstm_{i}"
            )(layer_input, initial_state=[inputs[1 + 2 * i], inputs[2 + 2 * i]])
            skip_connections.append(predictions)
            outputs.extend([state_h, state_c])

        final_outputs = []
        for i in range(self.num_layers):
            final_out = tf.keras.layers.Lambda(
                lambda x: x[:, -1, :],
                name=f"final_output_{i}"
            )(skip_connections[i])
            final_outputs.append(final_out)

        if self.num_layers == 1:
            layer_input_final = final_outputs[0]
        else:
            layer_input_final = tf.keras.layers.concatenate(
                final_outputs,
                name="final_concat"
            )
        dense = tf.keras.layers.Dense(self.dictionary.get_size(), name='dense_logits')(layer_input_final)
        output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(dense)

        self.model = tf.keras.Model(inputs=inputs, outputs=output)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.start_learning_rate),
            loss='sparse_categorical_crossentropy'
        )

    def _symbols_to_tokens(self, symbols, dictionary):
        """
        Convert a list of Symbol objects into a list of integer tokens using a sorted order.
        
        Args:
            symbols (list[Symbol]): A list of Symbol objects.
            dictionary (Dictionary): The dictionary containing Symbol objects.
            
        Returns:
            list[int]: A list of integer tokens.
            
        Raises:
            ValueError: If inputs are not valid.
        """
        if not isinstance(symbols, list):
            raise ValueError("symbols must be a list of Symbol objects")
        if not all(isinstance(s, Symbol) for s in symbols):
            raise ValueError("All elements in symbols must be instances of Symbol")
        if not hasattr(dictionary, 'symbols'):
            raise ValueError("dictionary must have a 'symbols' attribute")
        sorted_symbols = sorted(dictionary.symbols, key=lambda s: s.data)
        token_map = {symbol.data: idx for idx, symbol in enumerate(sorted_symbols)}
        tokens = []
        for symbol in symbols:
            token = token_map.get(symbol.data)
            if token is None:
                raise ValueError(f"Symbol with data {symbol.data} not found in dictionary.")
            tokens.append(token)
        return tokens

    def _preprocess_input(self, symbols, seq_length, batch_size=1, padding_value=0):
        """
        Convert a list of Symbol objects into a padded tensor for model input.
        If the input list is empty (i.e. empty context), returns a tensor of shape [batch_size, seq_length] filled with padding_value.
        
        Args:
            symbols (list[Symbol]): A list of Symbol objects.
            seq_length (int): The fixed sequence length for the model.
            batch_size (int): The batch size; must be a positive integer.
            padding_value (int): The token value used for padding.
            
        Returns:
            tf.Tensor: A tensor of shape [batch_size, seq_length].
            
        Raises:
            ValueError: If input is invalid.
        """
        import numpy as np
        from keras import utils

        if symbols is None or not isinstance(symbols, list):
            raise ValueError("symbols must be a list of Symbol objects")
        
        if len(symbols) == 0:
            return tf.fill([batch_size, seq_length], padding_value)
        
        tokens = self._symbols_to_tokens(symbols, self.dictionary)
        sequences = [tokens]
        padded_sequences = utils.pad_sequences(
            sequences, maxlen=seq_length, padding='post', truncating='post', value=padding_value
        )
        if padded_sequences.shape[0] != batch_size:
            padded_sequences = np.tile(padded_sequences, (batch_size, 1))
        return tf.convert_to_tensor(padded_sequences, dtype=tf.int32)

    def _postprocess_predictions(self, raw_output):
        """
        Convert the model's raw output tensor into a list of SymbolFrequency pairs.
        
        Args:
            raw_output (tf.Tensor or np.ndarray): The raw output with shape [batch_size, vocab_size].
            
        Returns:
            list[list[SymbolFrequency]]: A list (one per batch element) of SymbolFrequency pairs,
                                         sorted in descending order by frequency.
                                         
        Raises:
            ValueError: If raw_output is invalid.
        """
        if raw_output is None:
            raise ValueError("raw_output cannot be None")
        if isinstance(raw_output, tf.Tensor):
            raw_output = raw_output.numpy()
        if not isinstance(raw_output, np.ndarray):
            raise ValueError("raw_output must be a tf.Tensor or np.ndarray")
        sorted_symbols = sorted(self.dictionary.symbols, key=lambda s: s.data)
        batch_result = []
        for row in raw_output:
            symbol_freq_list = []
            for idx, prob in enumerate(row):
                try:
                    symbol = sorted_symbols[idx]
                except IndexError:
                    raise ValueError("Index out of range while mapping predictions to symbols.")
                symbol_freq_list.append(SymbolFrequency(symbol, prob))
            symbol_freq_list.sort(key=lambda sf: sf.frequency, reverse=True)
            batch_result.append(symbol_freq_list)
            
        return batch_result

    def predict(self, symbols):
        if symbols is None or not isinstance(symbols, list):
            raise ValueError("symbols must be a list of Symbol objects")

        main_input_tensor = self._preprocess_input(
            symbols, seq_length=self.seq_length, batch_size=self.batch_size
        )

        full_inputs = [main_input_tensor] + self.dummy_states

        raw_output = self.model(full_inputs, training=False)
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
        
        main_input_tensor = self._preprocess_input(
            previous_symbols, seq_length=self.seq_length, batch_size=self.batch_size
        )

        full_inputs = [main_input_tensor] + self.dummy_states

        target_tokens = self._symbols_to_tokens([correct_symbol], self.dictionary)
        target_token = target_tokens[0]
        target_tensor = tf.fill([self.batch_size], target_token)

        loss = self._train_step(full_inputs, target_tensor)
        tf.print("Training loss:", loss)


    def save_model(self, path):
        pass

    def load_model(self, path):
        pass

