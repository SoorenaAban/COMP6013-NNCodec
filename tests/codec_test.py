import unittest

from nncodec.codecs import *
from nncodec.coders import *
from nncodec.preprocessors import *

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

class TestTfCodec(unittest.TestCase):
    def test_compress_decompress(self):
        """Test that compressing and decompressing data preserves the original data."""
        codec = TfCodec()
        data = b'Testing Data'
        preprocessor = BytePreprocessor()
        coder = ArithmeticCoderOnline(ArithmeticCoderSettings())
        compressed_data = codec.compress(data, preprocessor, 0, coder)
        decompressed_data = codec.decompress(compressed_data)
        
        self.assertEqual(data, decompressed_data)
        
class TestTfCodecFile(unittest.TestCase):
    def test_compress_decompress(self):
        """Test that compressing and decompressing data preserves the original data."""
        codec = TfCodecFile()
        
        data = b'Testing Data'
        
        #create temporary file name contating the testing data
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(data)
        
        #perform compression
        
        compressed_file_name = temp_file_name + '.compressed'
        preprocessor = BytePreprocessor()
        coder = ArithmeticCoderOnline(ArithmeticCoderSettings())
        codec.compress(temp_file_name, compressed_file_name, preprocessor, 0, coder)
        
        #perform decompression
        
        decompressed_file_name = temp_file_name + '.decompressed'
        codec.decompress(compressed_file_name, decompressed_file_name)
        
        #read the decompressed data
        with open(decompressed_file_name, 'rb') as decompressed_file:
            decompressed_data = decompressed_file.read()
            
        self.assertEqual(data, decompressed_data)
        
        #remove temporary files
        os.remove(temp_file_name)
        os.remove(compressed_file_name)
        os.remove(decompressed_file_name)
        
class TestTfCodecByte(unittest.TestCase):
    def test_compress_decompress(self):
        """Test that compressing and decompressing data preserves the original data."""
        codec = TfCodecByte()
        data = b'Testing Data'
        compressed_data = codec.compress(data, 0, 1)
        decompressed_data = codec.decompress(compressed_data)
        
        self.assertEqual(data, decompressed_data)
        
class TestTfCodecByteArithmetic(unittest.TestCase):
    def test_compress_decompress(self):
        """Test that compressing and decompressing data preserves the original data."""
        codec = TfCodecByteArithmetic()
        data = b'Testing Data'
        compressed_data = codec.compress(data, 0, 2)
        decompressed_data = codec.decompress(compressed_data)
        
        self.assertEqual(data, decompressed_data)
        
class TestTfCodecByteArithmeticFile(unittest.TestCase):
    def test_compress_decompress(self):
        """Test that compressing and decompressing data preserves the original data."""
        codec = TfCodecByteArithmeticFile()
        
        data = b'Testing Data'
        
        #create temporary file name contating the testing data
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(data)
        
        #perform compression
        
        compressed_file_name = temp_file_name + '.compressed'
        codec.compress(temp_file_name, compressed_file_name, 0, 1)
        
        #perform decompression
        
        decompressed_file_name = temp_file_name + '.decompressed'
        codec.decompress(compressed_file_name, decompressed_file_name)
        
        #read the decompressed data
        with open(decompressed_file_name, 'rb') as decompressed_file:
            decompressed_data = decompressed_file.read()
            
        self.assertEqual(data, decompressed_data)
        
        #remove temporary files
        os.remove(temp_file_name)
        os.remove(compressed_file_name)
        os.remove(decompressed_file_name)
        
    def test_compress_decompress_deep(self):
        codec = TfCodecByteArithmeticFile()
        
        data = b'Testing Data'
        
        #create temporary file name contating the testing data
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(data)
        
        #perform compression
        
        compressed_file_name = temp_file_name + '.compressed'
        codec.compress(temp_file_name, compressed_file_name, 0, 2)
        
        #perform decompression
        
        decompressed_file_name = temp_file_name + '.decompressed'
        codec.decompress(compressed_file_name, decompressed_file_name)
        
        #read the decompressed data
        with open(decompressed_file_name, 'rb') as decompressed_file:
            decompressed_data = decompressed_file.read()
            
        self.assertEqual(data, decompressed_data)
        
        #remove temporary files
        os.remove(temp_file_name)
        os.remove(compressed_file_name)
        os.remove(decompressed_file_name)

if __name__ == '__main__':
    unittest.main()