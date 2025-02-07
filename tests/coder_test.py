import unittest
import numpy as np
import io
import sys

from nncodec import models
from nncodec import coder

class TestArithmeticCoder(unittest.TestCase):
    def setUp(self):
        self.dictionary = models.Dictionary()
        symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        self.dictionary.add_multiple(symbols)

        self.settings_override = {
            'ARITH_SCALING_FACTOR': 10000000,
            'ARITH_OFFSET': 1
        }
        self.coder = coder.arithmetic_coder(settings_override=self.settings_override)
    
    def test_probabilities_to_code(self):
        probs = [0.1, 0.2, 0.7]
        cum_freq = self.coder.probabilities_to_code(probs)
        expected_freqs = np.round(np.array(probs) * self.coder.scaling_factor + self.coder.offset).astype(np.int64)
        expected_cum_freq = np.cumsum(expected_freqs)
        np.testing.assert_array_equal(cum_freq, expected_cum_freq)

    def test_code_to_probabilities(self):
        cum_freq = np.array([2, 5, 10])
        probs = self.coder.code_to_probabilities(cum_freq)
        expected_probs = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(probs, expected_probs, atol=1e-5)
    
    def test_encode_returns_bytes(self):
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b')]
        result = self.coder.encode(input_symbols, self.dictionary, preprocessor=None, prediction_model=None)
        self.assertIsInstance(result, bytes)
    
    def test_decode_returns_symbols(self):
        encoded_data = b"encoded_data"
        result = self.coder.decode(encoded_data, self.dictionary, preprocessor=None, prediction_model=None)
        self.assertIsInstance(result, list)
        for sym in result:
            self.assertIsInstance(sym, models.Symbol)
    
    def test_input_validation_probabilities_to_code(self):
        with self.assertRaises(ValueError):
            self.coder.probabilities_to_code(None)
        with self.assertRaises(ValueError):
            # Sum not equal to 1.
            self.coder.probabilities_to_code([0.1, 0.1, 0.1])
    
    def test_input_validation_code_to_probabilities(self):
        with self.assertRaises(ValueError):
            self.coder.code_to_probabilities(None)
        with self.assertRaises(ValueError):
            self.coder.code_to_probabilities(np.array([[1, 2, 3], [4, 5, 6]]))

if __name__ == '__main__':
    unittest.main()