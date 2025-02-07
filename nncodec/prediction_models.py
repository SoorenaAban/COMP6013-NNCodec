#prediction-model.py

import abc
import tensorflow as tf
import numpy as np
print(tf.__version__)
from keras import mixed_precision
from keras import utils
from .models import Symbol, SymbolFrequency, Dictionary

class base_prediction_model(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def predict(self, symbols, dictionary):
        """returns the predicted data
        symbols: list of symbols to be used for prediction

        returns:
        """
        pass

    @abc.abstractmethod
    def train(self, in_symbols, correct_symbol):
        """trains the model"""
        pass

    @abc.abstractmethod
    def save_model(self, path):
        """saves the model to the given path"""
        pass

    @abc.abstractmethod
    def load_model(self, path):
        """loads the model from the given path"""
        pass

class tf_prediction_model(base_prediction_model):
    def __init__(self, dictionary, settings_override=None):
        """
        Initialize the TensorFlow prediction model.
        
        Args:
            dictionary (Dictionary): A Dictionary instance containing Symbol objects.
            settings_override (dict, optional): A dictionary with settings values to override the defaults.
                Expected keys: 
                    - 'TF_BATCH_SIZES'
                    - 'DEF_SEQ_LENGTH'
                    - 'TF_NUM_LAYERS'
                    - 'TF_RNN_UNITS'
                    - 'TF_EMBEDING_SIZE'
                    - 'TF_START_LEARNING_RATE'
                If not provided, default settings are loaded from the global 'settings' module.
        """
        self.dictionary = dictionary

        if settings_override is None:
            import settings  # assumes a settings module exists
            self._settings = {
                'TF_BATCH_SIZES': settings.TF_BATCH_SIZES,
                'DEF_SEQ_LENGTH': settings.DEF_SEQ_LENGTH,
                'TF_NUM_LAYERS': settings.TF_NUM_LAYERS,
                'TF_RNN_UNITS': settings.TF_RNN_UNITS,
                'TF_EMBEDING_SIZE': settings.TF_EMBEDING_SIZE,
                'TF_START_LEARNING_RATE': settings.TF_START_LEARNING_RATE,
            }
        else:
            self._settings = settings_override

        self.batch_size = self._settings['TF_BATCH_SIZES']
        self.seq_length = self._settings['DEF_SEQ_LENGTH']
        self.num_layers = self._settings['TF_NUM_LAYERS']
        self.rnn_units = self._settings['TF_RNN_UNITS']
        self.embedding_size = self._settings['TF_EMBEDING_SIZE']
        self.start_learning_rate = self._settings['TF_START_LEARNING_RATE']

        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

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
        )(embedded,initial_state=[inputs[1], inputs[2]]) #(embedded, initial_state=[tf.cast(inputs[1], tf.float16), tf.cast(inputs[2], tf.float16)])
        skip_connections = [predictions]
        outputs = [state_h, state_c]  # (Not used publicly for now)

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
            )(layer_input, initial_state=[inputs[1 + 2 * i], inputs[2 + 2 * i]])#(layer_input, initial_state=[tf.cast(inputs[1 + 2 * i], tf.float16),tf.cast(inputs[2 + 2 * i], tf.float16)])
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
            ValueError: If input is not valid.
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
        if not isinstance(symbols, list) or len(symbols) == 0:
            raise ValueError("symbols must be a non-empty list of Symbol objects")
        if not all(isinstance(s, Symbol) for s in symbols):
            raise ValueError("All items in symbols must be instances of Symbol")
        if not isinstance(seq_length, int) or seq_length <= 0:
            raise ValueError("seq_length must be a positive integer")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        tokens = self._symbols_to_tokens(symbols, self.dictionary)
        sequences = [tokens]  # Wrap as a single batch.
        padded_sequences = utils.pad_sequences(
            sequences, maxlen=seq_length, padding='post', truncating='post', value=padding_value
        )
        input_tensor = tf.convert_to_tensor(padded_sequences, dtype=tf.int32)
        return input_tensor

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
        """
        Preprocess input symbols, run the model, and return predictions.
        """
        if symbols is None or not isinstance(symbols, list) or len(symbols) == 0:
            raise ValueError("symbols must be a non-empty list of Symbol objects")

        main_input_tensor = self._preprocess_input(
            symbols, seq_length=self.seq_length, batch_size=self.batch_size
        )

        dummy_states = []
        for _ in range(self.num_layers):
            dummy_state = tf.zeros((self.batch_size, self.rnn_units), dtype=tf.float16)
            dummy_states.extend([dummy_state, dummy_state])

        full_inputs = [main_input_tensor] + dummy_states

        raw_output = self.model(full_inputs, training=False)
        predictions = self._postprocess_predictions(raw_output)
        return predictions[0]


    def train(self, previous_symbols, correct_symbol):
        """
        Train the model on a single example.
        """
        if previous_symbols is None or not isinstance(previous_symbols, list) or len(previous_symbols) == 0:
            raise ValueError("previous_symbols must be a non-empty list of Symbol objects")
        if not isinstance(correct_symbol, Symbol):
            raise ValueError("correct_symbol must be an instance of Symbol")

        main_input_tensor = self._preprocess_input(
            previous_symbols, seq_length=self.seq_length, batch_size=self.batch_size
        )

        dummy_states = []
        for _ in range(self.num_layers):
            dummy_state = tf.zeros((self.batch_size, self.rnn_units), dtype=tf.float16)
            dummy_states.extend([dummy_state, dummy_state])

        full_inputs = [main_input_tensor] + dummy_states

        target_tokens = self._symbols_to_tokens([correct_symbol], self.dictionary)
        target_token = target_tokens[0]
        target_tensor = tf.convert_to_tensor([target_token] * self.batch_size, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            predictions = self.model(full_inputs, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()(target_tensor, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        print(f"Training loss: {loss.numpy()}")


    def save_model(self, path):
        pass

    def load_model(self, path):
        pass

