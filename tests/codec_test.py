import unittest

from nncodec.codec import *

class TestCompressedModel(unittest.TestCase):
    def setUp(self):
        # Create a sample CompressedModel instance for testing.
        self.model = CompressedModel(
            preprocessor_code=3,
            version=1,
            preprocessor_header_size=1024,
            preprocessor_header=b'header_data',
            keras_code=2,
            coder_code=1,
            data=b'some_binary_data',
            original_file_name='test_file.txt'
        )
    
    def test_serialization_deserialization(self):
        """Test that a model can be serialized and then correctly deserialized."""
        serialized = CompressedModel.serialize(self.model)
        deserialized_model = CompressedModel.deserialize(serialized)
        
        self.assertEqual(self.model.preprocessor_code, deserialized_model.preprocessor_code)
        self.assertEqual(self.model.version, deserialized_model.version)
        self.assertEqual(self.model.preprocessor_header_size, deserialized_model.preprocessor_header_size)
        self.assertEqual(self.model.preprocessor_header, deserialized_model.preprocessor_header)
        self.assertEqual(self.model.keras_code, deserialized_model.keras_code)
        self.assertEqual(self.model.coder_code, deserialized_model.coder_code)
        self.assertEqual(self.model.data, deserialized_model.data)
        self.assertEqual(self.model.original_file_name, deserialized_model.original_file_name)
    
    def test_file_write_read(self):
        """Test that writing to and reading from a file preserves the model data."""
        # Use a temporary file for file I/O testing.
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name
        
        try:
            CompressedModelFile.write_to_file(self.model, temp_file_name)
            read_model = CompressedModelFile.read_from_file(temp_file_name)
            
            self.assertEqual(self.model.preprocessor_code, read_model.preprocessor_code)
            self.assertEqual(self.model.version, read_model.version)
            self.assertEqual(self.model.preprocessor_header_size, read_model.preprocessor_header_size)
            self.assertEqual(self.model.preprocessor_header, read_model.preprocessor_header)
            self.assertEqual(self.model.keras_code, read_model.keras_code)
            self.assertEqual(self.model.coder_code, read_model.coder_code)
            self.assertEqual(self.model.data, read_model.data)
            self.assertEqual(self.model.original_file_name, read_model.original_file_name)
        finally:
            os.remove(temp_file_name)

class TestTfCodecByte(unittest.TestCase):
    def test_compress_decompress(self):
        """Test that compressing and decompressing data preserves the original data."""
        codec = TfCodecByte()
        data = b'Testing Data'
        compressed_data = codec.compress(data, 0, 1)
        decompressed_data = codec.decompress(compressed_data)
        
        self.assertEqual(data, decompressed_data)

if __name__ == '__main__':
    unittest.main()