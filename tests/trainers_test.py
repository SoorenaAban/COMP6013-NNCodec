# trainers_test.py

import unittest
import tempfile
import os
import numpy as np

from nncodec import trainers, models
from nncodec.prediction_models import tf_prediction_model

class DummyTfPredictionModel(tf_prediction_model):
    def __init__(self, dictionary, settings_override, model_weights_path=None):
        super().__init__(dictionary, settings_override, model_weights_path)
        self.train_calls = []  

    def train(self, previous_symbols, correct_symbol):
        self.train_calls.append((previous_symbols, correct_symbol))

class TestTFTrainer(unittest.TestCase):
    def setUp(self):
        self.dictionary = models.Dictionary()
        symbols = [models.Symbol(b'a'), models.Symbol(b'b'),
                   models.Symbol(b'c'), models.Symbol(b'd')]
        self.dictionary.add_multiple(symbols)
        
        self.test_config = {
            'TF_SEED': 1234,
            'TF_BATCH_SIZES': 1,
            'TF_SEQ_LENGTH': 5,
            'TF_NUM_LAYERS': 2,
            'TF_RNN_UNITS': 16,
            'TF_EMBEDING_SIZE': 8,
            'TF_START_LEARNING_RATE': 0.001
        }
        self.trainer = trainers.tf_trainer(self.dictionary, self.test_config)

    def test_invalid_train_input(self):
        with self.assertRaises(ValueError):
            self.trainer.train("not a list")
        with self.assertRaises(ValueError):
            self.trainer.train([])

    def test_train_calls(self):
        dummy_model = DummyTfPredictionModel(self.dictionary, self.test_config)
        self.trainer.prediction_model = dummy_model
        
        training_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        self.trainer.train(training_symbols)
        
        self.assertEqual(len(dummy_model.train_calls), 3)
        
        prev, target = dummy_model.train_calls[0]
        self.assertEqual(prev, [])
        self.assertEqual(target.data, b'a')
        
        prev, target = dummy_model.train_calls[1]
        self.assertEqual(len(prev), 1)
        self.assertEqual(prev[0].data, b'a')
        self.assertEqual(target.data, b'b')
        
        prev, target = dummy_model.train_calls[2]
        self.assertEqual(len(prev), 2)
        self.assertEqual(prev[0].data, b'a')
        self.assertEqual(prev[1].data, b'b')
        self.assertEqual(target.data, b'c')

    def test_save_invalid_path(self):
        with self.assertRaises(ValueError):
            self.trainer.save("")
        with self.assertRaises(ValueError):
            self.trainer.save(123)

    def test_save_valid(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            self.trainer.save(tmp_path)
            expected_path = tmp_path if tmp_path.endswith(".weights.h5") else tmp_path + ".weights.h5"
            self.assertTrue(os.path.exists(expected_path))
        finally:
            if os.path.exists(expected_path):
                os.remove(expected_path)

if __name__ == '__main__':
    unittest.main()
