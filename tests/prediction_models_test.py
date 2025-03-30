import os
import tempfile
import unittest
from nncodec.models import Symbol, Dictionary, SymbolFrequency
from nncodec.prediction_models import TestingPredictionModel, TfPredictionModel
from nncodec.keras_models import get_keras_model
from nncodec.logger import Logger

class TestTestingPredictionModel(unittest.TestCase):
    def setUp(self):
        self.dictionary = Dictionary()
        self.dictionary.add(Symbol(b'a'))
        self.dictionary.add(Symbol(b'b'))
        self.dictionary.add(Symbol(b'c'))
        self.model = TestingPredictionModel(self.dictionary)
    
    def test_predict_returns_frequencies(self):
        context = [Symbol(b'a'), Symbol(b'b')]
        freqs = self.model.predict(context)
        self.assertEqual(len(freqs), self.dictionary.get_size())
        for sf in freqs:
            self.assertIsInstance(sf, SymbolFrequency)
    
    def test_train_does_not_error(self):
        context = [Symbol(b'a')]
        try:
            self.model.train(context, Symbol(b'b'))
        except Exception as e:
            self.fail(f"train() raised an exception: {e}")
    
    def test_save_and_load_model_no_error(self):
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_path = temp_file.name
            temp_file.close()
            self.model.save_model(temp_path)
            self.model.load_model(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

class TestTfPredictionModel(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.dictionary = Dictionary()
        for char in b"abcd":
            self.dictionary.add(Symbol(bytes([char])))
        self.keras_model = get_keras_model(0, self.dictionary.get_size())
        self.model = TfPredictionModel(self.dictionary, self.keras_model, logger=self.logger)
    
    def test_predict(self):
        context = [Symbol(b'a'), Symbol(b'b')]
        predictions = self.model.predict(context)
        self.assertIsInstance(predictions, list)
        for sf in predictions:
            self.assertIsInstance(sf, SymbolFrequency)
    
    def test_train(self):
        context = [Symbol(b'a'), Symbol(b'b')]
        try:
            self.model.train(context, Symbol(b'c'))
        except Exception as e:
            self.fail(f"TfPredictionModel.train raised an exception: {e}")
    
    def test_save_and_load_model(self):
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".weights.h5")
            temp_path = temp_file.name
            temp_file.close()
            self.model.save_model(temp_path)
            _ = self.model.predict([Symbol(b'a')])
            self.model.load_model(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
