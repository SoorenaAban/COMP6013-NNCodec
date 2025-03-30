import unittest
from nncodec.preprocessors import AsciiCharPreprocessor, BytePreprocessor, get_preprocessor
from nncodec.models import Symbol, Dictionary
from nncodec.logger import Logger

class TestAsciiCharPreprocessor(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.preprocessor = AsciiCharPreprocessor(self.logger)
    
    def test_convert_to_symbols_valid(self):
        data = b"ABC"
        symbols, dictionary = self.preprocessor.convert_to_symbols(data)
        expected_symbols = [Symbol(b'\x82'), Symbol(b'a'),
                            Symbol(b'\x82'), Symbol(b'b'),
                            Symbol(b'\x82'), Symbol(b'c')]
        self.assertEqual(symbols, expected_symbols)
        self.assertTrue(dictionary.contains_data(b'a'))
        self.assertTrue(dictionary.contains_data(b'b'))
        self.assertTrue(dictionary.contains_data(b'c'))
        self.assertTrue(dictionary.contains_data(b'\x82'))
    
    def test_convert_to_symbols_invalid(self):
        with self.assertRaises(ValueError):
            self.preprocessor.convert_to_symbols(b"\xff")
    
    def test_convert_from_symbols(self):
        data = b"ABC"
        symbols, _ = self.preprocessor.convert_to_symbols(data)
        reconstructed = self.preprocessor.convert_from_symbols(symbols)
        self.assertEqual(reconstructed, b"ABC")
    
    def test_encode_and_construct_dictionary(self):
        dictionary = Dictionary()
        dictionary.add(Symbol(b'a'))
        dictionary.add(Symbol(b'b'))
        dictionary.add(Symbol(b'\x82'))
        header = self.preprocessor.encode_dictionary_for_header(dictionary)
        self.assertEqual(len(header), 16)
        reconstructed_dict = self.preprocessor.construct_dictionary_from_header(header)
        self.assertTrue(reconstructed_dict.contains_data(b'a'))
        self.assertTrue(reconstructed_dict.contains_data(b'b'))
        self.assertTrue(reconstructed_dict.contains_data(b'\x82'))

class TestBytePreprocessor(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.preprocessor = BytePreprocessor(self.logger)
    
    def test_convert_to_symbols(self):
        data = b"\x01\x02\x01"
        symbols, dictionary = self.preprocessor.convert_to_symbols(data)
        expected_symbols = [Symbol(b'\x01'), Symbol(b'\x02'), Symbol(b'\x01')]
        self.assertEqual(symbols, expected_symbols)
        self.assertEqual(dictionary.get_size(), 2)
    
    def test_convert_from_symbols(self):
        data = b"\x10\x20\x30"
        symbols, _ = self.preprocessor.convert_to_symbols(data)
        reconstructed = self.preprocessor.convert_from_symbols(symbols)
        self.assertEqual(reconstructed, data)
    
    def test_encode_and_construct_dictionary(self):
        dictionary = Dictionary()
        dictionary.add(Symbol(b'\x05'))
        dictionary.add(Symbol(b'\x06'))
        header = self.preprocessor.encode_dictionary_for_header(dictionary)
        self.assertEqual(len(header), 32)
        reconstructed_dict = self.preprocessor.construct_dictionary_from_header(header)
        self.assertTrue(reconstructed_dict.contains_data(b'\x05'))
        self.assertTrue(reconstructed_dict.contains_data(b'\x06'))

class TestGetPreprocessor(unittest.TestCase):
    def test_get_byte_preprocessor(self):
        preprocessor = get_preprocessor(3)
        from nncodec.preprocessors import BytePreprocessor
        self.assertIsInstance(preprocessor, BytePreprocessor)
    
    def test_get_ascii_preprocessor(self):
        preprocessor = get_preprocessor(4)
        from nncodec.preprocessors import AsciiCharPreprocessor
        self.assertIsInstance(preprocessor, AsciiCharPreprocessor)
    
    def test_get_invalid_preprocessor(self):
        with self.assertRaises(ValueError):
            get_preprocessor(99)

if __name__ == '__main__':
    unittest.main()
