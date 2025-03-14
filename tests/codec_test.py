import unittest

from nncodec import codec

class TestNNcodecByteCodec(unittest.TestCase):
    def setUp(self):
        self.codec = codec.byte_codec()
        
    def test_compress_decompress(self):
        test_data = str.encode("Test")
        encoded_data = self.codec.compress(test_data)
        decoded_data = self.codec.decompress(encoded_data)
        self.assertEqual(test_data, decoded_data)

class TestNNcodecByteCodecDeep(unittest.TestCase):
    def setUp(self):
        self.codec = codec.byte_codec_deep()
        
    def test_compress_decompress(self):
        test_data = str.encode("Test")
        encoded_data = self.codec.compress(test_data)
        decoded_data = self.codec.decompress(encoded_data)
        self.assertEqual(test_data, decoded_data)
if __name__ == '__main__':
    unittest.main()