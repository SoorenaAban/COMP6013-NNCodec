import unittest
import numpy as np
import tensorflow as tf

from nncodec.keras_models import (
    LstmKerasModel,
    GruKerasModel,
    LstmKerasModelLight,
    TestingKerasModel
)

class KerasModelTestHelper:
    @staticmethod
    def build_dummy_input(model):
        main_input = tf.zeros((model.batch_size, model.seq_length), dtype=tf.int32)
        dummy_states = model.get_dummy_states()
        if dummy_states:
            return [main_input] + dummy_states
        else:
            return main_input

    @staticmethod
    def check_output(model, output):
        if output.shape[0] != model.batch_size:
            raise AssertionError(
                f"Expected batch size {model.batch_size}, got {output.shape[0]}"
            )
        if output.shape[1] != model.vocab_size:
            raise AssertionError(
                f"Expected vocab size {model.vocab_size}, got {output.shape[1]}"
            )
        row_sums = tf.reduce_sum(output, axis=1).numpy()
        np.testing.assert_allclose(row_sums, np.ones(model.batch_size), atol=1e-5)

class KerasModelsTest(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 10

    def test_TFPredictionLstmKerasModel(self):
        model = LstmKerasModel(vocab_size=self.vocab_size)
        dummy_input = KerasModelTestHelper.build_dummy_input(model)
        output = model(dummy_input, training=False)
        KerasModelTestHelper.check_output(model, output)

    def test_TFPredictionTestingKerasModel(self):
        model = TestingKerasModel(vocab_size=self.vocab_size)
        dummy_input = KerasModelTestHelper.build_dummy_input(model)
        output = model(dummy_input, training=False)
        KerasModelTestHelper.check_output(model, output)
        
    def test_TFPredictionGruKerasModel(self):
        model = GruKerasModel(vocab_size=self.vocab_size)
        dummy_input = KerasModelTestHelper.build_dummy_input(model)
        output = model(dummy_input, training=False)
        KerasModelTestHelper.check_output(model, output)
        
    def test_TFPredictionLstmKerasModelLight(self):
        model = LstmKerasModelLight(vocab_size=self.vocab_size)
        dummy_input = KerasModelTestHelper.build_dummy_input(model)
        output = model(dummy_input, training=False)
        KerasModelTestHelper.check_output(model, output)

if __name__ == '__main__':
    unittest.main()
