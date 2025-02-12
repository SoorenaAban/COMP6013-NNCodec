import unittest
import numpy as np
import io
import sys

from nncodec import models
from nncodec import coder
from nncodec import prediction_models as pred

# Dummy prediction model for testing arithmetic coder.
# class DummyPredictionModel:
#     def __init__(self, dictionary):
#         self.dictionary = dictionary

#     def predict(self, context):
#         """
#         Return a fixed probability distribution for testing.
#         For a dictionary of size N, returns a distribution with linearly increasing probabilities,
#         then normalized.
#         """
#         size = self.dictionary.get_size()
#         probs = np.linspace(1, size, num=size, dtype=np.float64)
#         probs = probs / np.sum(probs)
#         return probs

class TestArithmeticCoder(unittest.TestCase):
    def setUp(self):
        # Create a dummy dictionary with symbols: a, b, c.
        self.dictionary = models.Dictionary()
        symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        self.dictionary.add_multiple(symbols)
        
        # Settings override for arithmetic coder.
        self.settings_override = {
            'ARITH_SCALING_FACTOR': 10000000,
            'ARITH_OFFSET': 1
        }
        self.coder = coder.arithmetic_coder(settings_override=self.settings_override)
        self.prediction_model = pred.testing_prediction_model(self.dictionary)

    def test_probabilities_to_code(self):
        # Test with a valid probability distribution that sums to 1.
        probs = [0.1, 0.2, 0.7]
        cum_freq = self.coder.probabilities_to_code(probs)
        # Expected frequencies: round([0.1*scaling+offset, 0.2*scaling+offset, 0.7*scaling+offset])
        expected_freqs = np.round(np.array(probs) * self.coder.scaling_factor + self.coder.offset).astype(np.int64)
        expected_cum_freq = np.cumsum(expected_freqs)
        np.testing.assert_array_equal(cum_freq, expected_cum_freq)

    def test_code_to_probabilities(self):
        # Create a dummy cumulative frequency table.
        cum_freq = np.array([2, 5, 10])
        probs = self.coder.code_to_probabilities(cum_freq)
        expected_probs = np.array([2, 3, 5]) / 10.0
        np.testing.assert_allclose(probs, expected_probs, atol=1e-5)
    
    def test_invalid_probabilities_to_code(self):
        with self.assertRaises(ValueError):
            self.coder.probabilities_to_code(None)
        with self.assertRaises(ValueError):
            self.coder.probabilities_to_code([0.1, 0.1, 0.1])  # Does not sum to 1.
    
    def test_invalid_code_to_probabilities(self):
        with self.assertRaises(ValueError):
            self.coder.code_to_probabilities(None)
        with self.assertRaises(ValueError):
            # Passing a multidimensional array.
            self.coder.code_to_probabilities(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_encode_returns_bytes(self):
        # Create a simple list of symbols.
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        encoded_data = self.coder.encode(input_symbols, self.prediction_model)
        self.assertIsInstance(encoded_data, bytes)

    def test_decode_returns_symbols(self):
        # Encode some data first.
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        encoded_data = self.coder.encode(input_symbols, self.prediction_model)
        decoded_symbols = self.coder.decode(encoded_data, self.dictionary, self.prediction_model)
        self.assertIsInstance(decoded_symbols, list)
        for sym in decoded_symbols:
            self.assertIsInstance(sym, models.Symbol)

    def test_encode_decode(self):
        # Create a simple list of symbols.
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        encoded_data = self.coder.encode(input_symbols, self.prediction_model)
        decoded_symbols = self.coder.decode(encoded_data, self.dictionary, self.prediction_model)
        self.assertEqual(input_symbols, decoded_symbols)

    # def test_current_state_property(self):
    #     state = self.coder.current_state
    #     self.assertIn("low", state)
    #     self.assertIn("high", state)
    #     self.assertIn("state_bits", state)

if __name__ == '__main__':
    unittest.main()