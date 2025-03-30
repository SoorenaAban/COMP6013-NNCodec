import unittest
import numpy as np
import struct
import os
import types

from nncodec import models
from nncodec import coders
from nncodec import prediction_models as pred

class TestArithmeticCodec(unittest.TestCase):
    def setUp(self):
        self.settings = coders.ArithmeticCoderSettings()
        self.codec = coders.ArithmeticCodec(self.settings)
    
    def test_probabilities_to_code_valid(self):
        probs = [0.1, 0.2, 0.7]
        cum_freq = self.codec.probabilities_to_code(probs)
        expected_freqs = np.round(np.array(probs) * self.settings.scaling_factor + self.settings.offset).astype(np.int64)
        expected_cum_freq = np.cumsum(expected_freqs)
        np.testing.assert_array_equal(cum_freq, expected_cum_freq)
    
    def test_code_to_probabilities_valid(self):
        cum_freq = np.array([2, 5, 10])
        probs = self.codec.code_to_probabilities(cum_freq)
        expected_probs = np.array([2, 3, 5]) / 10.0
        np.testing.assert_allclose(probs, expected_probs, atol=1e-5)
    
    def test_probabilities_to_code_invalid_none(self):
        with self.assertRaises(ValueError):
            self.codec.probabilities_to_code(None)
    
    def test_probabilities_to_code_invalid_sum(self):
        with self.assertRaises(ValueError):
            self.codec.probabilities_to_code([0.1, 0.1, 0.1])
    
    def test_code_to_probabilities_invalid_none(self):
        with self.assertRaises(ValueError):
            self.codec.code_to_probabilities(None)
    
    def test_code_to_probabilities_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.codec.code_to_probabilities(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_pack_unpack_bits(self):
        bits = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0]
        packed = self.codec.pack_bits_to_bytes(bits)
        unpacked = self.codec.unpack_bytes_to_bits(packed)
        # Only compare the first len(bits) elements since padding might add extra zeros.
        np.testing.assert_array_equal(unpacked[:len(bits)], bits)

class TestArithmeticCoder(unittest.TestCase):
    def setUp(self):
        self.dictionary = models.Dictionary()
        symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        self.dictionary.add_multiple(symbols)
        
        self.arith_settings = coders.ArithmeticCoderSettings()
        self.coder = coders.ArithmeticCoderOnline(self.arith_settings)
        self.prediction_model = pred.testing_prediction_model(self.dictionary)

    def test_encode_returns_bytes(self):
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        encoded_data = self.coder.encode(input_symbols, self.prediction_model)
        self.assertIsInstance(encoded_data, bytes)

    def test_decode_returns_symbols(self):
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        encoded_data = self.coder.encode(input_symbols, self.prediction_model)
        decoded_symbols = self.coder.decode(encoded_data, self.dictionary, self.prediction_model)
        self.assertIsInstance(decoded_symbols, list)
        for sym in decoded_symbols:
            self.assertIsInstance(sym, models.Symbol)

    def test_encode_decode(self):
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        encoded_data = self.coder.encode(input_symbols, self.prediction_model)
        decoded_symbols = self.coder.decode(encoded_data, self.dictionary, self.prediction_model)
        self.assertEqual(input_symbols, decoded_symbols)

class TestArithmeticCoderDeep(unittest.TestCase):
    def setUp(self):
        self.dictionary = models.Dictionary()
        symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        self.dictionary.add_multiple(symbols)
        
        self.arith_settings = coders.ArithmeticCoderSettings()
        self.coder_deep = coders.ArithmeticCoderOffline(self.arith_settings)
        self.prediction_model = pred.testing_prediction_model(self.dictionary)
        
        def dummy_save_model(self, path):
            with open(path, "wb") as f:
                f.write(b"dummy_weights")
        
        def dummy_load_model(self, path):
            with open(path, "rb") as f:
                data = f.read()
            self.loaded_weights = data
        
        self.prediction_model.save_model = types.MethodType(dummy_save_model, self.prediction_model)
        self.prediction_model.load_model = types.MethodType(dummy_load_model, self.prediction_model)

    def test_encode_returns_bytes(self):
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        encoded_data = self.coder_deep.encode(input_symbols, self.prediction_model)
        self.assertIsInstance(encoded_data, bytes)
    
    def test_weights_header(self):
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        encoded_data = self.coder_deep.encode(input_symbols, self.prediction_model)
        header = encoded_data[:8]
        weights_length = struct.unpack(">Q", header)[0]
        self.assertEqual(weights_length, 13)
        self.assertGreaterEqual(len(encoded_data), 8 + weights_length)
    
    def test_decode_returns_symbols(self):
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        encoded_data = self.coder_deep.encode(input_symbols, self.prediction_model)
        decoded_symbols = self.coder_deep.decode(encoded_data, self.dictionary, self.prediction_model)
        self.assertIsInstance(decoded_symbols, list)
        for sym in decoded_symbols:
            self.assertIsInstance(sym, models.Symbol)
    
    def test_encode_decode(self):
        input_symbols = [models.Symbol(b'a'), models.Symbol(b'b'), models.Symbol(b'c')]
        encoded_data = self.coder_deep.encode(input_symbols, self.prediction_model)
        decoded_symbols = self.coder_deep.decode(encoded_data, self.dictionary, self.prediction_model)
        self.assertEqual(input_symbols, decoded_symbols)

if __name__ == '__main__':
    unittest.main()
