# trainers_test.py

import unittest
import tempfile
import os

from nncodec import trainers, models
from nncodec.keras_models import TestingKerasModel


class TestTFTrainer(unittest.TestCase):
    def setUp(self):
        self.dictionary = models.Dictionary()
        self.test_keras_model = TestingKerasModel(4)
        symbols = [models.Symbol(b'a'), models.Symbol(b'b'),
                   models.Symbol(b'c'), models.Symbol(b'd')]
        self.dictionary.add_multiple(symbols)
        
        self.trainer = trainers.tf_trainer(self.dictionary, self.test_keras_model)

    def test_invalid_train_input(self):
        with self.assertRaises(ValueError):
            self.trainer.train("not a list")
        with self.assertRaises(ValueError):
            self.trainer.train([])

    def test_train(self):
        symbols_collection = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        self.trainer.train(symbols_collection)
        assert self.trainer.prediction_model.model is not None
        

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
