import unittest


from nncodec import preprocessors
from nncodec import models

class TestBytePreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = preprocessors.byte_preprocessor()
        self.data = b"hello world"
        self.symbols, self.dictionary = self.preprocessor.convert_to_symbols(self.data)

    def test_convert_to_symbols(self):
        self.assertIsInstance(self.symbols, list)
        self.assertIsInstance(self.dictionary, models.Dictionary)
        self.assertEqual(len(self.symbols), len(self.data))

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
        self.assertEqual(dictionary.get_size(), len(set(self.symbols)))

    def test_encode_dictionary_for_header(self):
        encoded = self.preprocessor.encode_dictionary_for_header(self.dictionary)
        self.assertIsInstance(encoded, bytes)