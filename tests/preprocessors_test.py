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

class TestAsciiCharPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = preprocessors.ascii_char_preprocessor()

    def test_convert_to_symbols_lowercase(self):
        data = b"hello"
        symbols, dictionary = self.preprocessor.convert_to_symbols(data)
        self.assertEqual(len(symbols), len(data))
        expected_dict_size = len(set(data)) + 1
        self.assertEqual(dictionary.get_size(), expected_dict_size)

    def test_convert_to_symbols_with_uppercase(self):
        data = b"TeSt"
        symbols, dictionary = self.preprocessor.convert_to_symbols(data)
        expected_uppercase_count = sum(1 for byte in data if chr(byte).isupper())
        self.assertEqual(len(symbols), len(data) + expected_uppercase_count)
        expected_letters = {chr(b).lower().encode('ascii') for b in data}
        expected_dict = expected_letters.union({b'\x82'})
        actual_dict = {symbol.data for symbol in dictionary.symbols}
        self.assertEqual(actual_dict, expected_dict)

    def test_convert_from_symbols(self):
        data = b"TeSt"
        symbols, _ = self.preprocessor.convert_to_symbols(data)
        result = self.preprocessor.convert_from_symbols(symbols)
        self.assertEqual(result, data)

    def test_invalid_non_ascii_input(self):
        with self.assertRaises(ValueError):
            self.preprocessor.convert_to_symbols(b"hello\xff")

    def test_invalid_data_type(self):
        with self.assertRaises(ValueError):
            self.preprocessor.convert_to_symbols("hello")

    def test_header_size_property(self):
        self.assertEqual(self.preprocessor.header_size, 16)

    def test_code_property(self):
        self.assertEqual(self.preprocessor.code, 4)

    def test_empty_data(self):
        data = b""
        symbols, dictionary = self.preprocessor.convert_to_symbols(data)
        self.assertEqual(symbols, [])
        self.assertEqual(dictionary.get_size(), 1)
        result = self.preprocessor.convert_from_symbols(symbols)
        self.assertEqual(result, b"")

    def test_encode_dictionary_for_header(self):
        data = b"test"
        _, dictionary = self.preprocessor.convert_to_symbols(data)
        encoded = self.preprocessor.encode_dictionary_for_header(dictionary)
        self.assertEqual(len(encoded), self.preprocessor.header_size)
        header_int = int.from_bytes(encoded, 'little')
        self.assertEqual(header_int >> 128, 0)

    def test_construct_dictionary_from_header(self):
        header_int = (1 << 97) | (1 << 98)
        header = header_int.to_bytes(16, 'little')
        dictionary = self.preprocessor.construct_dictionary_from_header(header)
        expected = {chr(97).encode('ascii'), chr(98).encode('ascii'), b'\x82'}
        actual = {symbol.data for symbol in dictionary.symbols}
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
