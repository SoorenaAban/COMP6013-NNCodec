#keras_models.py

import tensorflow as tf
from keras import layers

from .settings import *

class TFKerasModelBase(tf.keras.Model):
    """
    Base class for pre-defined Keras models in our prediction framework.
    Provides placeholder methods for serialization/deserialization, a helper
    to generate dummy states for LSTM layers, and enforces determinism by setting
    a global seed during initialization.
    """
    def __init__(self, **kwargs):
        super(TFKerasModelBase, self).__init__(**kwargs)
        self.seed = TF_SEED

    def serialize_model(self):
        pass

    def deserialize_model(self, serialized_data):
        pass

    def get_dummy_states(self):
        """
        Returns a list of dummy state tensors for all LSTM layers.
        Assumes the derived model has attributes: num_layers, batch_size, and rnn_units.
        For each LSTM layer, returns a pair of zeros (for hidden and cell states).
        """
        dummy_states = []
        if hasattr(self, 'num_layers') and hasattr(self, 'batch_size') and hasattr(self, 'rnn_units'):
            for _ in range(self.num_layers):
                state = tf.zeros((self.batch_size, self.rnn_units), dtype=tf.float32)
                dummy_states.extend([state, state])
        return dummy_states


class LstmKerasModel(TFKerasModelBase):
    """ The parameters of this Lstm Keras subclass are based on the tensorflow-compress library"""
    def __init__(self, vocab_size, **kwargs):
        super(LstmKerasModel, self).__init__(**kwargs)
        self.batch_size = 256
        self.seq_length = 15
        self.num_layers = 6
        self.rnn_units = 1000
        self.embedding_size = 1024
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=self.seed),
            name="embedding"
        )
        
        self.lstm_layers = []
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
        def call(self, inputs):
            return inputs[:, -1, :]

    @tf.function
    def call(self, inputs, training=False):
        expected_inputs = 1 + 2 * self.num_layers
        if not isinstance(inputs, (list, tuple)) or len(inputs) != expected_inputs:
            raise ValueError(f"Expected {expected_inputs} inputs, got {len(inputs)}")
        
        main_input = inputs[0]
        embedded = self.embedding(main_input)
        skip_connections = []
        
        init_state_0 = [inputs[1], inputs[2]]
        x, _, _ = self.lstm_layers[0](embedded, initial_state=init_state_0, training=training)
        skip_connections.append(x)
        
        for i in range(1, self.num_layers):
            concatenated = tf.keras.layers.concatenate(
                [embedded, skip_connections[i-1]],
                name=f"concat_{i}"
            )
            init_state = [inputs[1 + 2*i], inputs[2 + 2*i]]
            x, _, _ = self.lstm_layers[i](concatenated, initial_state=init_state, training=training)
            skip_connections.append(x)
        
        final_outputs = [self.last_time_steps[i](skip_connections[i]) for i in range(self.num_layers)]
        if self.num_layers == 1:
            layer_input_final = final_outputs[0]
        else:
            layer_input_final = tf.keras.layers.concatenate(final_outputs, name="final_concat")
        logits = self.dense(layer_input_final)
        return self.softmax(logits)

import tensorflow as tf

class GruKerasModel(TFKerasModelBase):
    """GRU-based Keras subclass model mimicking the LSTM model architecture."""
    def __init__(self, vocab_size, **kwargs):
        super(GruKerasModel, self).__init__(**kwargs)
        self.batch_size = 256
        self.seq_length = 15
        self.num_layers = 6
        self.rnn_units = 1000
        self.embedding_size = 1024
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=self.seed),
            name="embedding"
        )

        self.gru_layers = []
        for i in range(self.num_layers):
            gru_layer = tf.keras.layers.GRU(
                units=self.rnn_units,
                return_sequences=True,
                return_state=True,
                # activation='tanh',
                # recurrent_activation='sigmoid',
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
        def call(self, inputs):
            return inputs[:, -1, :]

    @tf.function
    def call(self, inputs, training=False):
        """
        Expected inputs: a list of 1 + 2 * num_layers tensors.
          - inputs[0]: the main input (token sequences)
          - For each GRU layer, two dummy states are provided (for compatibility),
            but only the first is used.
        """
        expected_inputs = 1 + 2 * self.num_layers
        if not isinstance(inputs, (list, tuple)) or len(inputs) != expected_inputs:
            raise ValueError(f"Expected {expected_inputs} inputs, got {len(inputs)}")

        main_input = inputs[0]
        embedded = self.embedding(main_input)
        skip_connections = []

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


class TFPredictionTestingKerasModel(TFKerasModelBase):
    def __init__(self, vocab_size, **kwargs):
        super(TFPredictionTestingKerasModel, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.batch_size = 1
        self.seq_length = 5
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
    
    def call(self, inputs, training=False):
        if isinstance(inputs, (list, tuple)):
            main_input = inputs[0]
        else:
            main_input = inputs
        x = self.embedding(main_input)
        x = self.pool(x)
        logits = self.dense(x)
        return self.softmax(logits)
    
    def get_dummy_states(self):
        return []

class SimpleGRUModel(TFKerasModelBase):
    def __init__(self, vocab_size, **kwargs):
        super(SimpleGRUModel, self).__init__(**kwargs)
        self.batch_size = 256
        self.seq_length = 15
        self.num_layers = 1 
        self.rnn_units = 512
        self.embedding_size = 512
        self.vocab_size = vocab_size

        self.embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=self.seed),
            name="embedding"
        )
        self.gru = layers.GRU(
            units=self.rnn_units,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=False, 
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed+1),
            name="gru"
        )
        self.dense = layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed+2),
            name="dense_logits"
        )
        self.softmax = layers.Activation("softmax", dtype="float32", name="predictions")
    
    def call(self, inputs, training=False):
        main_input = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        x = self.embedding(main_input)
        x = self.gru(x, training=training)
        logits = self.dense(x)
        return self.softmax(logits)
    
    def get_dummy_states(self):
        return []


class StackedGRUModel(TFKerasModelBase):
    def __init__(self, vocab_size, **kwargs):
        super(StackedGRUModel, self).__init__(**kwargs)
        self.batch_size = 256
        self.seq_length = 15
        self.num_layers = 3
        self.rnn_units = 768
        self.embedding_size = 768
        self.vocab_size = vocab_size

        self.embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=self.seed),
            name="embedding"
        )
        self.gru_layers = []
        for i in range(self.num_layers):
            return_seq = True if i < self.num_layers - 1 else False
            gru = layers.GRU(
                units=self.rnn_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=return_seq,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + i),
                recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + self.num_layers + i),
                name=f"gru_{i}"
            )
            self.gru_layers.append(gru)
        self.dropout = layers.Dropout(rate=0.2, name="dropout")
        self.dense = layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 2*self.num_layers),
            name="dense_logits"
        )
        self.softmax = layers.Activation("softmax", dtype="float32", name="predictions")
    
    def call(self, inputs, training=False):
        main_input = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        x = self.embedding(main_input)
        for i, gru in enumerate(self.gru_layers):
            x = gru(x, training=training)
            if i < self.num_layers - 1:
                x = self.dropout(x, training=training)
        logits = self.dense(x)
        return self.softmax(logits)
    
    def get_dummy_states(self):
        return []


class BidirectionalGRUModel(TFKerasModelBase):
    def __init__(self, vocab_size, **kwargs):
        super(BidirectionalGRUModel, self).__init__(**kwargs)
        self.batch_size = 256
        self.seq_length = 15
        self.num_layers = 1 
        self.rnn_units = 512
        self.embedding_size = 512
        self.vocab_size = vocab_size

        self.embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=self.seed),
            name="embedding"
        )
        gru = layers.GRU(
            units=self.rnn_units,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=False,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed+1),
            name="gru"
        )
        self.bidirectional = layers.Bidirectional(gru, name="bidirectional_gru")
        self.dense = layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed+2),
            name="dense_logits"
        )
        self.softmax = layers.Activation("softmax", dtype="float32", name="predictions")
    
    def call(self, inputs, training=False):
        main_input = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        x = self.embedding(main_input)
        x = self.bidirectional(x, training=training)
        logits = self.dense(x)
        return self.softmax(logits)
    
    def get_dummy_states(self):
        return []

class DeepResidualGRUModel(TFKerasModelBase):
    def __init__(self, vocab_size, **kwargs):
        super(DeepResidualGRUModel, self).__init__(**kwargs)
        self.batch_size = 256
        self.seq_length = 15
        self.num_layers = 4
        self.rnn_units = 1024
        self.embedding_size = 1024
        self.vocab_size = vocab_size

        self.embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=self.seed),
            name="embedding"
        )
        self.gru_layers = []
        for i in range(self.num_layers):
            gru = layers.GRU(
                units=self.rnn_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=True,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + i),
                recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + self.num_layers + i),
                name=f"gru_{i}"
            )
            self.gru_layers.append(gru)
        self.dense = layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 2 * self.num_layers),
            name="dense_logits"
        )
        self.softmax = layers.Activation("softmax", dtype="float32", name="predictions")
    
    def call(self, inputs, training=False):
        main_input = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        x = self.embedding(main_input)
        for gru in self.gru_layers:
            y = gru(x, training=training)
            if x.shape[-1] == y.shape[-1]:
                x = x + y
            else:
                x = y
        last_output = x[:, -1, :]
        logits = self.dense(last_output)
        return self.softmax(logits)
    
    def get_dummy_states(self):
        return []