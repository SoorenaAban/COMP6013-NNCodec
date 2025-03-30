import os
import tempfile
import unittest
from nncodec.codecs import (
    CompressedModel,
    CompressedModelFile,
    enable_determinism,
    TfCodec,
    TfCodecFile,
    TfCodecByte,
    TfCodecByteArithmetic,
    TfCodecByteArithmeticFile
)
from nncodec.preprocessors import BytePreprocessor
from nncodec.coders import get_coder
from nncodec.keras_models import get_keras_model
from nncodec.logger import Logger

class TestCompressedModel(unittest.TestCase):
    def setUp(self):
        self.model = CompressedModel(
            preprocessor_code=3,
            version=1,
            preprocessor_header_size=32,
            preprocessor_header=b'header_data',
            keras_code=0, 
            coder_code=1,
            data=b'some_binary_data',
            original_file_name='test_file.txt'
        )
    
    def test_serialization_deserialization(self):
        serialized = CompressedModel.serialize(self.model)
        deserialized = CompressedModel.deserialize(serialized)
        self.assertEqual(self.model.preprocessor_code, deserialized.preprocessor_code)
        self.assertEqual(self.model.version, deserialized.version)
        self.assertEqual(self.model.preprocessor_header_size, deserialized.preprocessor_header_size)
        self.assertEqual(self.model.preprocessor_header, deserialized.preprocessor_header)
        self.assertEqual(self.model.keras_code, deserialized.keras_code)
        self.assertEqual(self.model.coder_code, deserialized.coder_code)
        self.assertEqual(self.model.data, deserialized.data)
        self.assertEqual(self.model.original_file_name, deserialized.original_file_name)
    
    def test_file_write_read(self):
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

class TestTfCodecVariants(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.keras_code = 0
        self.coder_online = get_coder(1, self.logger)
        self.coder_offline = get_coder(2, self.logger)
        self.data = b"Testing Data"
    
    def test_TfCodecByte_compress_decompress(self):
        codec = TfCodecByte()
        compressed_model = codec.compress(self.data, self.keras_code, 1)
        decompressed = codec.decompress(compressed_model)
        self.assertEqual(self.data, decompressed)
    
    def test_TfCodecByteArithmetic_compress_decompress(self):
        codec = TfCodecByteArithmetic()
        compressed_model = codec.compress(self.data, self.keras_code, offline_learning=True)
        decompressed = codec.decompress(compressed_model)
        self.assertEqual(self.data, decompressed)
    
    def test_TfCodecFile_compress_decompress(self):
        codec = TfCodecFile()
        with tempfile.NamedTemporaryFile(delete=False) as temp_input:
            input_file = temp_input.name
            temp_input.write(self.data)
        compressed_file = input_file + ".compressed"
        decompressed_file = input_file + ".decompressed"
        try:
            codec.compress(input_file, compressed_file, BytePreprocessor(), self.keras_code, get_coder(1, self.logger))
            codec.decompress(compressed_file, decompressed_file)
            with open(decompressed_file, "rb") as f:
                decompressed_data = f.read()
            self.assertEqual(self.data, decompressed_data)
        finally:
            for f in [input_file, compressed_file, decompressed_file]:
                if os.path.exists(f):
                    os.remove(f)
    
    def test_TfCodecByteArithmeticFile_compress_decompress(self):
        codec = TfCodecByteArithmeticFile()
        with tempfile.NamedTemporaryFile(delete=False) as temp_input:
            input_file = temp_input.name
            temp_input.write(self.data)
        compressed_file = input_file + ".compressed"
        decompressed_file = input_file + ".decompressed"
        try:
            codec.compress(input_file, compressed_file, self.keras_code, 2)
            codec.decompress(compressed_file, decompressed_file)
            with open(decompressed_file, "rb") as f:
                decompressed_data = f.read()
            self.assertEqual(self.data, decompressed_data)
        finally:
            for f in [input_file, compressed_file, decompressed_file]:
                if os.path.exists(f):
                    os.remove(f)

class TestEnableDeterminism(unittest.TestCase):
    def test_invalid_seed(self):
        with self.assertRaises(ValueError):
            enable_determinism(None)
    
    def test_valid_seed(self):
        try:
            enable_determinism(42)
        except Exception as e:
            self.fail(f"enable_determinism raised an exception with a valid seed: {e}")

if __name__ == '__main__':
    unittest.main()
