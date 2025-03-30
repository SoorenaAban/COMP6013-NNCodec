import tensorflow as tf
from typing import Any, List, Union

from .settings import SEED 


class TFKerasModelBase(tf.keras.Model):
    """     Base class for predefined Keras models in our prediction framework. """
    def __init__(self, **kwargs: Any) -> None:
        super(TFKerasModelBase, self).__init__(**kwargs)
        self.seed: int = SEED
        self.keras_code: int = -1

    def serialize_model(self) -> None:
        """Serialize the model. To be implemented by subclasses."""
        pass

    def deserialize_model(self, serialized_data: bytes) -> None:
        """Deserialize the model from bytes. To be implemented by subclasses."""
        pass

    def get_dummy_states(self) -> List[tf.Tensor]:
        """
        Returns a list of dummy state tensors for all RNN layers.
        For each RNN layer, returns a pair of zeros (for hidden and cell states).
        
        Returns:
            List[tf.Tensor]: Dummy state tensors.
        """
        dummy_states: List[tf.Tensor] = []
        if hasattr(self, 'num_layers') and hasattr(self, 'batch_size') and hasattr(self, 'rnn_units'):
            for _ in range(self.num_layers):
                state = tf.zeros((self.batch_size, self.rnn_units), dtype=tf.float32)
                dummy_states.extend([state, state])
        return dummy_states


class LstmKerasModel(TFKerasModelBase):
    """
    LSTM-based Keras model.
    
    Uses stateful behavior by returning updated LSTM states.
    """
    def __init__(self, vocab_size: int, **kwargs: Any) -> None:
        super(LstmKerasModel, self).__init__(**kwargs)
        self.batch_size: int = 256
        self.seq_length: int = 15
        self.num_layers: int = 6
        self.rnn_units: int = 1000
        self.embedding_size: int = 1024
        self.vocab_size: int = vocab_size
        self.keras_code: int = 1

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=self.seed),
            name="embedding"
        )
        
        self.lstm_layers: List[tf.keras.layers.LSTM] = []
        for i in range(self.num_layers):
            lstm_layer = tf.keras.layers.LSTM(
                units=self.rnn_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=True,
                return_state=True,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + i),
                recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + self.num_layers + i),
                name=f"lstm_{i}"
            )
            self.lstm_layers.append(lstm_layer)
        
        self.last_time_steps = [
            LstmKerasModel.LastTimeStep(name=f"final_output_{i}")
            for i in range(self.num_layers)
        ]
        
        self.dense = tf.keras.layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 2 * self.num_layers),
            name='dense_logits'
        )
        self.softmax = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')

    class LastTimeStep(tf.keras.layers.Layer):
        def call(self, inputs: tf.Tensor) -> tf.Tensor:
            return inputs[:, -1, :]

    @tf.function
    def call(self, inputs: Union[List[tf.Tensor], tuple], training: bool = False, return_states: bool = False) -> Union[tf.Tensor, List[tf.Tensor]]:
        """
        Process input tensors through the LSTM model.
        
        Args:
            inputs (list or tuple of tf.Tensor): contains 1 + 2 * num_layers tensors.
                - inputs[0]: Main input tensor.
                - For each LSTM layer, two tensors for initial hidden and cell states.
            training (bool): Whether in training mode.
            return_states (bool): If True, also return the updated LSTM states.
        
        Returns:
            tf.Tensor or list of tf.Tensor: The output predictions tensor, optionally with updated states.
        
        Raises:
            ValueError: If the number of inputs does not match the expected count.
        """
        expected_inputs = 1 + 2 * self.num_layers
        if not isinstance(inputs, (list, tuple)) or len(inputs) != expected_inputs:
            raise ValueError(f"Expected {expected_inputs} inputs, got {len(inputs)}")
        
        main_input = inputs[0]
        embedded = self.embedding(main_input)
        skip_connections: List[tf.Tensor] = []
        new_states: List[tf.Tensor] = []
        
        init_state_0 = [inputs[1], inputs[2]]
        x, state_h, state_c = self.lstm_layers[0](embedded, initial_state=init_state_0, training=training)
        skip_connections.append(x)
        new_states.extend([state_h, state_c])
        
        for i in range(1, self.num_layers):
            concatenated = tf.keras.layers.concatenate(
                [embedded, skip_connections[i-1]],
                name=f"concat_{i}"
            )
            init_state = [inputs[1 + 2 * i], inputs[2 + 2 * i]]
            x, state_h, state_c = self.lstm_layers[i](concatenated, initial_state=init_state, training=training)
            skip_connections.append(x)
            new_states.extend([state_h, state_c])
        
        final_outputs = [self.last_time_steps[i](skip_connections[i]) for i in range(self.num_layers)]
        if self.num_layers == 1:
            layer_input_final = final_outputs[0]
        else:
            layer_input_final = tf.keras.layers.concatenate(final_outputs, name="final_concat")
        logits = self.dense(layer_input_final)
        predictions = self.softmax(logits)
        
        if return_states:
            return [predictions] + new_states
        else:
            return predictions


class GruKerasModel(TFKerasModelBase):
    """
    GRU-based Keras model mimicking the LSTM model architecture.
    """
    def __init__(self, vocab_size: int, **kwargs: Any) -> None:
        super(GruKerasModel, self).__init__(**kwargs)
        self.batch_size: int = 256
        self.seq_length: int = 15
        self.num_layers: int = 6
        self.rnn_units: int = 1000
        self.embedding_size: int = 1024
        self.vocab_size: int = vocab_size
        self.keras_code: int = 2

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=self.seed),
            name="embedding"
        )

        self.gru_layers: List[tf.keras.layers.GRU] = []
        for i in range(self.num_layers):
            gru_layer = tf.keras.layers.GRU(
                units=self.rnn_units,
                return_sequences=True,
                return_state=True,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + i),
                recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + self.num_layers + i),
                name=f"gru_{i}"
            )
            self.gru_layers.append(gru_layer)

        self.last_time_steps = [
            GruKerasModel.LastTimeStep(name=f"final_output_{i}")
            for i in range(self.num_layers)
        ]

        self.dense = tf.keras.layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 2 * self.num_layers),
            name='dense_logits'
        )
        self.softmax = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')

    class LastTimeStep(tf.keras.layers.Layer):
        def call(self, inputs: tf.Tensor) -> tf.Tensor:
            return inputs[:, -1, :]

    @tf.function
    def call(self, inputs: Union[List[tf.Tensor], tuple], training: bool = False) -> tf.Tensor:
        """
        Process input tensors through the GRU model.
        
        Expected inputs: a list of 1 + 2 * num_layers tensors.
          - inputs[0]: the main input (token sequences)
          - For each GRU layer, a single initial state is used.
        
        Args:
            inputs (list or tuple of tf.Tensor): Input tensors.
            training (bool): Whether in training mode.
        
        Returns:
            tf.Tensor: The predictions tensor.
        
        Raises:
            ValueError: If the number of inputs does not match the expected count.
        """
        expected_inputs = 1 + 2 * self.num_layers
        if not isinstance(inputs, (list, tuple)) or len(inputs) != expected_inputs:
            raise ValueError(f"Expected {expected_inputs} inputs, got {len(inputs)}")

        main_input = inputs[0]
        embedded = self.embedding(main_input)
        skip_connections: List[tf.Tensor] = []

        init_state_0 = inputs[1]  
        x, _ = self.gru_layers[0](embedded, initial_state=[init_state_0], training=training)
        skip_connections.append(x)

        for i in range(1, self.num_layers):
            concatenated = tf.keras.layers.concatenate(
                [embedded, skip_connections[i-1]],
                name=f"concat_{i}"
            )
            init_state = inputs[1 + 2 * i]  
            x, _ = self.gru_layers[i](concatenated, initial_state=[init_state], training=training)
            skip_connections.append(x)

        final_outputs = [self.last_time_steps[i](skip_connections[i]) for i in range(self.num_layers)]
        if self.num_layers == 1:
            layer_input_final = final_outputs[0]
        else:
            layer_input_final = tf.keras.layers.concatenate(final_outputs, name="final_concat")
        logits = self.dense(layer_input_final)
        return self.softmax(logits)


class LstmKerasModelLight(TFKerasModelBase):
    """
    Lightweight LSTM-based Keras model.
    
    Parameters are based on the tensorflow-compress library.
    """
    def __init__(self, vocab_size: int, **kwargs: Any) -> None:
        super(LstmKerasModelLight, self).__init__(**kwargs)
        self.batch_size: int = 256
        self.seq_length: int = 15
        self.num_layers: int = 4
        self.rnn_units: int = 512
        self.embedding_size: int = 512
        self.vocab_size: int = vocab_size
        self.keras_code: int = 3

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=self.seed),
            name="embedding"
        )
        
        self.embedding_proj = tf.keras.layers.Dense(self.rnn_units, name="embedding_proj")
        
        self.lstm_layers: List[tf.keras.layers.LSTM] = []
        for i in range(self.num_layers):
            lstm_layer = tf.keras.layers.LSTM(
                units=self.rnn_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=True,
                return_state=True,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + i),
                recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + self.num_layers + i),
                name=f"lstm_{i}"
            )
            self.lstm_layers.append(lstm_layer)
        
        self.last_time_steps = [
            LstmKerasModelLight.LastTimeStep(name=f"final_output_{i}")
            for i in range(self.num_layers)
        ]
        
        self.dense = tf.keras.layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 2 * self.num_layers),
            name='dense_logits'
        )
        self.softmax = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')

    class LastTimeStep(tf.keras.layers.Layer):
        """"Layer to extract the last time step from the LSTM output."""
        def call(self, inputs: tf.Tensor) -> tf.Tensor:
            """
            Extract the last time step from the input tensor.
            
            Args:
                inputs (tf.Tensor): The input tensor.
            
            Returns:
                tf.Tensor: The last time step of the input tensor.
            """    
            
            return inputs[:, -1, :]

    @tf.function
    def call(self, inputs: Union[List[tf.Tensor], tuple], training: bool = False) -> tf.Tensor:
        """
        Process input tensors through the lightweight LSTM model.
        
        Args:
            inputs (list or tuple of tf.Tensor): Expected to contain 1 + 2 * num_layers tensors.
            training (bool): Whether in training mode.
        
        Returns:
            tf.Tensor: The predictions tensor.
        
        Raises:
            ValueError: If the number of inputs does not match the expected count.
        """
        expected_inputs = 1 + 2 * self.num_layers
        if not isinstance(inputs, (list, tuple)) or len(inputs) != expected_inputs:
            raise ValueError(f"Expected {expected_inputs} inputs, got {len(inputs)}")
        
        main_input = inputs[0]
        embedded = self.embedding(main_input)
        embedded_proj = self.embedding_proj(embedded)
        skip_connections: List[tf.Tensor] = []
        
        init_state_0 = [inputs[1], inputs[2]]
        x, _, _ = self.lstm_layers[0](embedded, initial_state=init_state_0, training=training)
        skip_connections.append(x)
        
        for i in range(1, self.num_layers):
            x_input = embedded_proj + skip_connections[i-1]
            init_state = [inputs[1 + 2*i], inputs[2 + 2*i]]
            x, _, _ = self.lstm_layers[i](x_input, initial_state=init_state, training=training)
            skip_connections.append(x)
        
        final_outputs = [self.last_time_steps[i](skip_connections[i]) for i in range(self.num_layers)]
        if self.num_layers == 1:
            layer_input_final = final_outputs[0]
        else:
            layer_input_final = tf.keras.layers.concatenate(final_outputs, name="final_concat")
        logits = self.dense(layer_input_final)
        return self.softmax(logits)


class TestingKerasModel(TFKerasModelBase):
    """ Testing Keras model. Used for unit tests"""
    def __init__(self, vocab_size: int, **kwargs: Any) -> None:
        super(TestingKerasModel, self).__init__(**kwargs)
        self.keras_code: int = 0
        self.vocab_size: int = vocab_size
        self.batch_size: int = 1
        self.seq_length: int = 5
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=64,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=self.seed),
            name="embedding"
        )
        self.pool = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")
        self.dense = tf.keras.layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 1),
            name="dense_logits"
        )
        self.softmax = tf.keras.layers.Activation("softmax", dtype="float32", name="predictions")
    
    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]], training: bool = False) -> tf.Tensor:
        """
        Process input tensors through the testing model.
        
        Args:
            inputs (tf.Tensor or list of tf.Tensor): The input tensor(s).
            training (bool): Whether in training mode.
        
        Returns:
            tf.Tensor: The predictions tensor.
        """
        if isinstance(inputs, (list, tuple)):
            main_input = inputs[0]
        else:
            main_input = inputs
        x = self.embedding(main_input)
        x = self.pool(x)
        logits = self.dense(x)
        return self.softmax(logits)
    
    def get_dummy_states(self) -> List[tf.Tensor]:
        """Testing model does not use dummy states."""
        return []


def get_keras_model(code: int, vocab_size: int) -> TFKerasModelBase:
    """
    Retrieve a Keras model instance based on the given code.
    
    Args:
        code (int): The Keras model code.
        vocab_size (int): The vocabulary size.
    
    Returns:
        TFKerasModelBase: An instance of a Keras model.
    
    Raises:
        ValueError: If the model code is unsupported.
    """
    if code == 0:
        return TestingKerasModel(vocab_size)
    elif code == 1:
        return LstmKerasModel(vocab_size)
    elif code == 2:
        return GruKerasModel(vocab_size)
    elif code == 3:
        return LstmKerasModelLight(vocab_size)
    else:
        raise ValueError(f"Unsupported Keras model code: {code}")
