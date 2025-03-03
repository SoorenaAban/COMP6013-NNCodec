# prediction_models_test.py

import unittest
import numpy as np
import sys
import io
import os
import tempfile

from nncodec import prediction_models
from nncodec import models

test_settings = {
    'TF_SEED': 1234,
    'TF_BATCH_SIZES': 1,
    'TF_SEQ_LENGTH': 5,
    'TF_NUM_LAYERS': 2,
    'TF_RNN_UNITS': 16,
    'TF_EMBEDING_SIZE': 8,
    'TF_START_LEARNING_RATE': 0.001
}

class TestTFPredictionModel(unittest.TestCase):
    def setUp(self):
        self.dictionary = models.Dictionary()
        symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c'),
                   models.Symbol(b'd'), models.Symbol(b'e')]
        self.dictionary.add_multiple(symbols)

        self.model_obj = prediction_models.tf_prediction_model(self.dictionary, settings_override=test_settings)
        self.input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        self.correct_symbol = models.Symbol(b'd')

    def test_symbols_to_tokens(self):
        tokens = self.model_obj._symbols_to_tokens(self.input_symbols, self.dictionary)
        self.assertEqual(tokens, [0, 1, 2])

    def test_preprocess_input(self):
        input_tensor = self.model_obj._preprocess_input(self.input_symbols,
                                                        seq_length=test_settings['TF_SEQ_LENGTH'],
                                                        batch_size=test_settings['TF_BATCH_SIZES'])
        self.assertEqual(input_tensor.shape, (test_settings['TF_BATCH_SIZES'], test_settings['TF_SEQ_LENGTH']))

    def test_postprocess_predictions(self):
        vocab_size = self.dictionary.get_size()
        dummy_output = np.array([[0.1, 0.3, 0.25, 0.2, 0.15]])
        processed = self.model_obj._postprocess_predictions(dummy_output)
        self.assertEqual(len(processed), test_settings['TF_BATCH_SIZES'])
        top_sf = processed[0][0]
        self.assertIsInstance(top_sf, models.SymbolFrequency)
        self.assertEqual(top_sf.symbol.data, b'b')

    def test_predict(self):
        predictions = self.model_obj.predict(self.input_symbols)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), self.dictionary.get_size())

    def test_train(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        self.model_obj.train(self.input_symbols, self.correct_symbol)
        sys.stdout = sys.__stdout__  # Reset redirect.

    def test_save_model_invalid_path(self):
        with self.assertRaises(ValueError):
            self.model_obj.save_model("")
        with self.assertRaises(ValueError):
            self.model_obj.save_model(123)

    def test_load_model_invalid_path(self):
        with self.assertRaises(ValueError):
            self.model_obj.load_model("")
        with self.assertRaises(ValueError):
            self.model_obj.load_model("non_existent_file")

    def test_save_and_load_model(self):
        original_weights = self.model_obj.model.get_weights()
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            self.model_obj.save_model(tmp_path)
            
            new_weights = [np.full_like(w, 7) for w in self.model_obj.model.get_weights()]
            self.model_obj.model.set_weights(new_weights)
            
            modified_weights = self.model_obj.model.get_weights()
            for orig, mod in zip(original_weights, modified_weights):
                self.assertFalse(np.array_equal(orig, mod))
            
            self.model_obj.load_model(tmp_path)
            restored_weights = self.model_obj.model.get_weights()
            
            for orig, restored in zip(original_weights, restored_weights):
                self.assertTrue(np.array_equal(orig, restored))
        finally:
            os.remove(tmp_path + ".weights.h5" if not tmp_path.endswith(".weights.h5") else tmp_path)

    def test_constructor_with_weights(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            self.model_obj.save_model(tmp_path)
            new_model_obj = prediction_models.tf_prediction_model(self.dictionary,
                                                                    settings_override=test_settings,
                                                                    model_weights_path=tmp_path)
            original_weights = self.model_obj.model.get_weights()
            new_weights = new_model_obj.model.get_weights()
            for orig, new in zip(original_weights, new_weights):
                self.assertTrue(np.array_equal(orig, new))
        finally:
            os.remove(tmp_path + ".weights.h5" if not tmp_path.endswith(".weights.h5") else tmp_path)

if __name__ == '__main__':
    unittest.main()
