#codec.py

import abc
import os
import random
import numpy as np
import tensorflow as tf
from keras import mixed_precision

from .prediction_models import *
from .preprocessors import *
from .coder import *
from .models import *
from .keras_models import *
from .settings import *
from .logger import *

class CompressedModel:
    """To be used to represent the compressed file"""
    def __init__(self, preprocessor_code, version, preprocessor_header_size, preprocessor_header, keras_code, coder_code, data, original_file_name=None):
        if not isinstance(preprocessor_code, int):
            raise ValueError("Preprocessor code should be of type int")
        try:
            preprocessor = get_preprocessor(preprocessor_code)
        except ValueError:
            raise ValueError("Preprocessor code is not valid")
        if not isinstance(keras_code, int):
            raise ValueError("Keras code should be of type int")
        # try:
        #     keras_model = get_keras_model(keras_code)
        # except ValueError:
        #     raise ValueError("Keras code is not valid")
        if not isinstance(coder_code, int):
            raise ValueError("Coder code should be of type int")
        try:
            coder = get_coder(coder_code)
        except ValueError:
            raise ValueError("Coder code is not valid")
        if not isinstance(version, int):
            raise ValueError("Version should be of type int")
        if version != 1:
            raise ValueError("Version not supported")
        if not isinstance(preprocessor_header_size, int):
            raise ValueError("Preprocessor size should be of type int")
        if not isinstance(preprocessor_header, bytes):
            raise ValueError("Preprocessor header should be of type bytes")
        if not isinstance(data, bytes):
            raise ValueError("Data should be of type bytes")
        if len(preprocessor_header) > preprocessor_header_size:
            raise ValueError("Preprocessor header size should be less than preprocessor size")
        if preprocessor_header_size < 0:
            raise ValueError("Preprocessor size should be greater than 0")
        if original_file_name is not None and not isinstance(original_file_name, str):
            raise ValueError("Original file name should be of type str")
        
        self.original_file_name = original_file_name
        self.preprocessor_code = preprocessor_code
        self.version = version
        self.preprocessor_header_size = preprocessor_header_size
        self.preprocessor_header = preprocessor_header
        self.keras_code = keras_code
        self.coder_code = coder_code
        self.data = data

    @staticmethod
    def serialize(model: 'CompressedModel') -> bytes:
        """
        Serializes a CompressedModel instance into bytes.
        
        The format:
          - preprocessor_code (4 bytes, unsigned int)
          - version (4 bytes, unsigned int)
          - preprocessor_header_size (4 bytes, unsigned int)
          - actual header length (4 bytes, unsigned int)
          - preprocessor_header (variable length, actual header length bytes)
          - keras_code (4 bytes, unsigned int)
          - coder_code (4 bytes, unsigned int)
          - data length (4 bytes, unsigned int)
          - data (variable length)
          - original_file_name length (4 bytes, unsigned int; 0 if None)
          - original_file_name (UTF-8 encoded, if present)
        """
        header_actual_len = len(model.preprocessor_header)
        file_name_bytes = model.original_file_name.encode('utf-8') if model.original_file_name is not None else b''
        file_name_length = len(file_name_bytes)
        
        serialized = struct.pack('IIII', 
                                 model.preprocessor_code, 
                                 model.version, 
                                 model.preprocessor_header_size, 
                                 header_actual_len)
        serialized += model.preprocessor_header
        serialized += struct.pack('II', model.keras_code, model.coder_code)
        serialized += struct.pack('I', len(model.data))
        serialized += model.data
        serialized += struct.pack('I', file_name_length)
        serialized += file_name_bytes
        
        return serialized

    @staticmethod
    def deserialize(serialized: bytes) -> 'CompressedModel':
        """
        Deserializes bytes into a CompressedModel instance.
        The byte structure is expected to be the same as produced by serialize().
        """
        if len(serialized) < 16:
            raise ValueError("Serialized data is too short")
        preprocessor_code, version, preprocessor_header_size, header_actual_len = struct.unpack('IIII', serialized[:16])
        offset = 16
        
        preprocessor_header = serialized[offset:offset+header_actual_len]
        offset += header_actual_len
        
        if len(serialized) < offset + 4:
            raise ValueError("Serialized data is incomplete for keras_code")
        keras_code, = struct.unpack('I', serialized[offset:offset+4])
        offset += 4

        if len(serialized) < offset + 4:
            raise ValueError("Serialized data is incomplete for coder_code")
        coder_code, = struct.unpack('I', serialized[offset:offset+4])
        offset += 4
        
        if len(serialized) < offset + 4:
            raise ValueError("Serialized data is incomplete for data length")
        data_length, = struct.unpack('I', serialized[offset:offset+4])
        offset += 4
        data = serialized[offset:offset+data_length]
        offset += data_length
        
        if len(serialized) < offset + 4:
            raise ValueError("Serialized data is incomplete for file name length")
        file_name_length, = struct.unpack('I', serialized[offset:offset+4])
        offset += 4
        if file_name_length > 0:
            original_file_name = serialized[offset:offset+file_name_length].decode('utf-8')
        else:
            original_file_name = None

        return CompressedModel(preprocessor_code, version, preprocessor_header_size, preprocessor_header, keras_code, coder_code, data, original_file_name)


class CompressedModelFile:
    """Static class for writing and reading a CompressedModel instance to/from a file."""
    
    @staticmethod
    def write_to_file(model: CompressedModel, file_path: str):
        """
        Serializes the model and writes it as binary data to the given file.
        """
        serialized_data = CompressedModel.serialize(model)
        with open(file_path, 'wb') as f:
            f.write(serialized_data)

    @staticmethod
    def read_from_file(file_path: str) -> CompressedModel:
        """
        Reads binary data from the given file and deserializes it into a CompressedModel instance.
        """
        with open(file_path, 'rb') as f:
            serialized_data = f.read()
        return CompressedModel.deserialize(serialized_data)

def enable_determinism(seed):
        """
        Initialize random seeds for determinism.
        
        Args:
            seed (int): The seed value.
        """
        if seed is None:
            raise ValueError("seed cannot be None")
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        mixed_precision.set_global_policy('mixed_float16')

class TfCodec:
    
    
    def compress(self, data, preprocessor, keras_model_code, coder, logger = None):
        """
        Compresses the input data.

        Args:
            data (list[bytes]): The list of bytes representing the data
            
        Returns:
            CompressedModel: The compressed model
        """
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        
        if not isinstance(preprocessor, BasePreprocessor):
            raise ValueError("Preprocessor should be of type base_preprocessor")
        
        if not isinstance(keras_model_code, int):
            raise ValueError("Keras model code should be of type int")
        
        if not isinstance(coder, CoderBase):
            raise ValueError("Coder should be of type base_coder")
        
        
        prpr_code = preprocessor.code
        syms, dictionary = preprocessor.convert_to_symbols(data)
        prpr_header = preprocessor.encode_dictionary_for_header(dictionary)
        keras_model = get_keras_model(keras_model_code, dictionary.get_size())
        predicitor = TfPredictionModel(dictionary, keras_model)
        data = coder.encode(syms, predicitor)
        compressed_model = CompressedModel(prpr_code, 1, preprocessor.header_size, prpr_header, keras_model_code, coder.get_coder_code(), data)
        return compressed_model
    
    def decompress(self, compressed_model, logger = None):
        """
        Decompresses encoded data.

        Args:
            compressed_data (CompressedModel): The compressed model
            
        Returns:
            list[bytes]: The list of bytes representing the decoded data
        """
        if not isinstance(compressed_model, CompressedModel):
            raise ValueError("Data should be in form of CompressedModel")
        
        if compressed_model.version != 1:
            raise ValueError("Version not supported")
        
        preprocessor = get_preprocessor(compressed_model.preprocessor_code)
        if preprocessor is None:
            raise ValueError("Preprocessor not supported")
        
        dictionary = preprocessor.construct_dictionary_from_header(compressed_model.preprocessor_header)
        if dictionary is None:
            raise ValueError("Dictionary not supported")
        
        keras_model = get_keras_model(compressed_model.keras_code, dictionary.get_size())
        if keras_model is None:
            raise ValueError("Keras model not supported")
        
        coder = get_coder(compressed_model.coder_code)
        if coder is None:
            raise ValueError("Coder not supported")
        
        predictor = TfPredictionModel(dictionary, keras_model)
        syms = coder.decode(compressed_model.data, dictionary, predictor)
        return preprocessor.convert_from_symbols(syms)


class TfCodecByte(TfCodec):
    def compress(self, data, keras_model_code, coder_code, logger = None):
        """
        Compresses the input data.

        Args:
            data (list[bytes]): The list of bytes representing the data
            
        Returns:
            CompressedModel: The compressed model
        """
        coder = get_coder(coder_code)
        
        
        return super().compress(data, BytePreprocessor(), keras_model_code, coder, logger)

class TfCodecByteArithmetic(TfCodecByte):
    def compress(self, data, keras_model_code, used_deep = True, logger = None):
        """
        Compresses the input data.

        Args:
            data (list[bytes]): The list of bytes representing the data
            
        Returns:
            CompressedModel: The compressed model
        """
        
        coder_settings = ArithmeticCoderSettings()
        
        if used_deep:
            coder = ArithmeticCoderDeep(coder_settings, logger)
        else:
            coder = ArithmeticCoder(coder_settings, logger)
        
        return super().compress(data, keras_model_code, coder, logger)