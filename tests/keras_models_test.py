import unittest
import numpy as np
import tensorflow as tf

from nncodec.keras_models import (
    TFKerasModelBase,
    LstmKerasModel,
    TFPredictionTestingKerasModel,
    SimpleGRUModel,
    StackedGRUModel,
    BidirectionalGRUModel,
    DeepResidualGRUModel
)

class KerasModelsTest(unittest.TestCase):
    def setUp(self):
        # Use a small vocabulary for testing.
        self.vocab_size = 10

    def build_dummy_input(self, model):
        """
        Build dummy input for the given model.
        If model.get_dummy_states() returns non-empty, assume the model expects a list:
        [main_input, ...dummy_states]. Otherwise, return a single tensor.
        """
        main_input = tf.zeros((model.batch_size, model.seq_length), dtype=tf.int32)
        dummy_states = model.get_dummy_states()
        if dummy_states:
            return [main_input] + dummy_states
        else:
            return main_input

    def check_output(self, model, output):
        """
        Verify that the output tensor has shape [batch_size, vocab_size] and that
        each row sums to 1 (i.e. a valid softmax probability distribution).
        """
        self.assertEqual(output.shape[0], model.batch_size)
        self.assertEqual(output.shape[1], model.vocab_size)
        row_sums = tf.reduce_sum(output, axis=1).numpy()
        np.testing.assert_allclose(row_sums, np.ones(model.batch_size), atol=1e-5)

    def test_TFPredictionDefaultKerasModel(self):
        model = LstmKerasModel(vocab_size=self.vocab_size)
        dummy_input = self.build_dummy_input(model)
        output = model(dummy_input)
        self.check_output(model, output)

    def test_TFPredictionTestingKerasModel(self):
        model = TFPredictionTestingKerasModel(vocab_size=self.vocab_size)
        dummy_input = self.build_dummy_input(model)
        output = model(dummy_input)
        self.check_output(model, output)

    def test_SimpleGRUModel(self):
        model = SimpleGRUModel(vocab_size=self.vocab_size)
        dummy_input = self.build_dummy_input(model)
        output = model(dummy_input)
        self.check_output(model, output)

    def test_StackedGRUModel(self):
        model = StackedGRUModel(vocab_size=self.vocab_size)
        dummy_input = self.build_dummy_input(model)
        output = model(dummy_input)
        self.check_output(model, output)

    def test_BidirectionalGRUModel(self):
        model = BidirectionalGRUModel(vocab_size=self.vocab_size)
        dummy_input = self.build_dummy_input(model)
        output = model(dummy_input)
        self.check_output(model, output)

    def test_DeepResidualGRUModel(self):
        model = DeepResidualGRUModel(vocab_size=self.vocab_size)
        dummy_input = self.build_dummy_input(model)
        output = model(dummy_input)
        self.check_output(model, output)

if __name__ == '__main__':
    unittest.main()
