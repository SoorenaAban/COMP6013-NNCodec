import unittest

from nncodec import preprocessors
from nncodec import models

class TestBytePreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = preprocessors.byte_preprocessor()
        self.data = b"Test Data"
        self.symbols, self.dictionary = self.preprocessor.convert_to_symbols(self.data)

    def test_convert_to_symbols(self):
        self.assertIsInstance(self.symbols, list)
        self.assertIsInstance(self.dictionary, models.Dictionary)
        self.assertEqual(len(self.symbols), len(self.data))
        self.assertEqual(self.dictionary.get_size(), len(set(self.data)))

    def test_convert_from_symbols(self):
        converted_data = self.preprocessor.convert_from_symbols(self.symbols)
        self.assertEqual(converted_data, self.data)

    def test_convert_to_and_from_symbols(self):
        symbols, _ = self.preprocessor.convert_to_symbols(self.data)
        converted_data = self.preprocessor.convert_from_symbols(symbols)
        self.assertEqual(converted_data, self.data)

    def test_construct_dictionary_from_symbols(self):
        dictionary = self.preprocessor.construct_dictionary_from_symbols(self.symbols)
        self.assertIsInstance(dictionary, models.Dictionary)
        self.assertEqual(dictionary.get_size(), len(set(self.data)))

    def test_encode_dictionary_for_header(self):
        encoded = self.preprocessor.encode_dictionary_for_header(self.dictionary)
        self.assertIsInstance(encoded, bytes)
        self.assertEqual(len(encoded), self.preprocessor.header_size)

    def test_dictionary_header_roundtrip(self):
        encoded = self.preprocessor.encode_dictionary_for_header(self.dictionary)
        reconstructed_dictionary = self.preprocessor.construct_dictionary_from_header(encoded)
        original_bytes = {symbol.data for symbol in self.dictionary.symbols}
        reconstructed_bytes = {symbol.data for symbol in reconstructed_dictionary.symbols}
        self.assertEqual(original_bytes, reconstructed_bytes)

    def test_header_size_property(self):
        self.assertEqual(self.preprocessor.header_size, 32)

    def test_empty_data(self):
        data = b""
        symbols, dictionary = self.preprocessor.convert_to_symbols(data)
        self.assertEqual(symbols, [])
        self.assertEqual(dictionary.get_size(), 0)
        encoded = self.preprocessor.encode_dictionary_for_header(dictionary)
        self.assertEqual(encoded, b'\x00' * 32)
        reconstructed_dictionary = self.preprocessor.construct_dictionary_from_header(encoded)
        self.assertEqual(reconstructed_dictionary.get_size(), 0)

if __name__ == '__main__':
    unittest.main()
