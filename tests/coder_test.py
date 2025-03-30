import unittest
from io import BytesIO
import numpy as np

from nncodec.coders import (
    ArithmeticCoderSettings,
    BitOutputStream,
    BitInputStream,
    ArithmeticCodec,
    ArithmeticCoderOnline,
    ArithmeticCoderOffline,
    get_coder
)
from nncodec.models import Symbol, Dictionary
from nncodec.prediction_models import TfPredictionModel
from nncodec.logger import Logger
from nncodec.keras_models import get_keras_model

class TestArithmeticCoderSettings(unittest.TestCase):
    def test_settings_attributes(self):
        settings = ArithmeticCoderSettings(scaling_factor=5000000, offset=2)
        self.assertEqual(settings.scaling_factor, 5000000)
        self.assertEqual(settings.offset, 2)

class TestBitStreamHelpers(unittest.TestCase):
    def test_bit_output_stream(self):
        out = BytesIO()
        bos = BitOutputStream(out)
        bits = [1, 0, 1, 0, 1, 0, 1, 0]
        for bit in bits:
            bos.write(bit)
        bos.finish()
        result = out.getvalue()
        self.assertEqual(result, bytes([0b10101010]))
    
    def test_bit_output_stream_padding(self):
        out = BytesIO()
        bos = BitOutputStream(out)
        for bit in [1, 0, 1]:
            bos.write(bit)
        bos.finish()
        result = out.getvalue()
        self.assertEqual(result, bytes([0b10100000]))
    
    def test_bit_input_stream(self):
        data = bytes([0b11001010])
        inp = BytesIO(data)
        bis = BitInputStream(inp)
        bits = [bis.read() for _ in range(8)]
        self.assertEqual(bits, [1, 1, 0, 0, 1, 0, 1, 0])
    
    def test_invalid_bit_write(self):
        out = BytesIO()
        bos = BitOutputStream(out)
        with self.assertRaises(ValueError):
            bos.write(2)

class TestArithmeticCodecHelpers(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.settings = ArithmeticCoderSettings()
        self.codec = ArithmeticCodec(self.settings, self.logger)
    
    def test_probabilities_to_code(self):
        probs = [0.5, 0.5]
        cum_freq = self.codec.probabilities_to_code(probs)
        freq = round(0.5 * self.settings.scaling_factor + self.settings.offset)
        expected = np.array([freq, freq * 2], dtype=np.int64)
        np.testing.assert_array_equal(cum_freq, expected)
    
    def test_code_to_probabilities(self):
        cum_freq = np.array([5000001, 10000002], dtype=np.int64)
        probs = self.codec.code_to_probabilities(cum_freq)
        self.assertAlmostEqual(probs[0], 0.5, places=5)
        self.assertAlmostEqual(probs[1], 0.5, places=5)
    
    def test_pack_and_unpack_bits(self):
        bits = [1, 0, 1, 0, 1, 0, 1, 0, 1]  # 9 bits.
        packed = self.codec.pack_bits_to_bytes(bits)
        self.assertEqual(packed, bytes([0b10101010, 0b10000000]))
        unpacked = self.codec.unpack_bytes_to_bits(packed)
        self.assertEqual(unpacked[:9], bits)

class TestArithmeticCoders(unittest.TestCase):
    def setUp(self):
        self.symbols = [Symbol(b'a'), Symbol(b'b'), Symbol(b'c')]
        self.dictionary = Dictionary()
        self.dictionary.add_multiple(self.symbols)
        self.logger = Logger()
        self.pred_model = TfPredictionModel(self.dictionary, get_keras_model(0, self.dictionary.get_size()))
    
    def test_online_coder_encode_decode(self):
        coder = ArithmeticCoderOnline(ArithmeticCoderSettings(), self.logger)
        encoded = coder.encode(self.symbols, self.pred_model)
        decoded = coder.decode(encoded, self.dictionary, self.pred_model)
        self.assertEqual(decoded, self.symbols)
    
    def test_offline_coder_encode_decode(self):
        coder = ArithmeticCoderOffline(ArithmeticCoderSettings(), self.logger)
        encoded = coder.encode(self.symbols, self.pred_model)
        decoded = coder.decode(encoded, self.dictionary, self.pred_model)
        self.assertEqual(decoded, self.symbols)
    
    def test_get_coder(self):
        coder_online = get_coder(1, self.logger)
        self.assertEqual(coder_online.get_coder_code(), 1)
        coder_offline = get_coder(2, self.logger)
        self.assertEqual(coder_offline.get_coder_code(), 2)
        with self.assertRaises(ValueError):
            get_coder(99, self.logger)

if __name__ == '__main__':
    unittest.main()
