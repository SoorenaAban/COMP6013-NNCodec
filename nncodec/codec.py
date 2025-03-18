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


class base_codec(abc.ABC):
    """The base class for the codecs."""
    @abc.abstractmethod
    def compress(self, data):
        """
        Compresses the input data.

        Args:
            data (list[bytes]): The list of bytes representing the data
            
        Returns:
            CompressedModel: The compressed model
        """
        pass
    @abc.abstractmethod
    def decompress(self, compressed_model):
        """
        Decompresses encoded data.

        Args:
            compressed_data (CompressedModel): The compressed model
            
        Returns:
            list[bytes]: The list of bytes representing the decoded data
        """
        pass

class BaseCodec:
    
    
    def compress(self, data, preprocessor, predictor, coder):
        """
        Compresses the input data.

        Args:
            data (list[bytes]): The list of bytes representing the data
            
        Returns:
            CompressedModel: The compressed model
        """
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        
        if not isinstance(preprocessor, base_preprocessor):
            raise ValueError("Preprocessor should be of type base_preprocessor")
        
        if not isinstance(predictor, base_prediction_model):
            raise ValueError("Predictor should be of type base_prediction_model")
        
        if not isinstance(coder, CoderBase):
            raise ValueError("Coder should be of type base_coder")
        
        compressed_model = CompressedModel()
        compressed_model.version = 1
        compressed_model.vocab_code = preprocessor.code
        syms, dictionary = preprocessor.convert_to_symbols(data)
        compressed_model.preprocessor_header = preprocessor.encode_dictionary_for_header(dictionary)
        compressed_model.data = coder.encode(syms, predictor)
        return compressed_model
    
    def decompress(self, compressed_model, preprocessor, predictor, coder):
        """
        Decompresses encoded data.

        Args:
            compressed_data (CompressedModel): The compressed model
            
        Returns:
            list[bytes]: The list of bytes representing the decoded data
        """
        if not isinstance(compressed_model, CompressedModel):
            raise ValueError("Data should be in form of CompressedModel")
        
        if not isinstance(preprocessor, base_preprocessor):
            raise ValueError("Preprocessor should be of type base_preprocessor")
        
        if not isinstance(predictor, base_prediction_model):
            raise ValueError("Predictor should be of type base_prediction_model")
        
        if not isinstance(coder, CoderBase):
            raise ValueError("Coder should be of type base_coder")
        
        if compressed_model.version != 1:
            raise ValueError("Version not supported")
        
        if compressed_model.vocab_code != preprocessor.code:
            raise ValueError("Preprocessor not supported")
        
        data = compressed_model.data
        dictionary = preprocessor.construct_dictionary_from_header(compressed_model.preprocessor_header)
        syms = coder.decode(data, dictionary, predictor)
        return preprocessor.convert_from_symbols(syms)


class ByteCodec(BaseCodec):
    pass

class byte_codec(base_codec):
    
    def __init__(self):
        enable_determinism(TF_SEED)
    
    def compress(self, data):
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        compressed_data = CompressedModel()
        compressed_data.version = 1
        prpr = byte_preprocessor()
        compressed_data.vocab_code = prpr.code
        syms, dictionary = prpr.convert_to_symbols(data)
        compressed_data.preprocessor_header = prpr.encode_dictionary_for_header(dictionary)
        keras_model = SimpleGRUModel(dictionary.get_size())
        predictor = TfPredictionModel(dictionary, keras_model)
        coder_settings = ArithmeticCoderSettings()
        codr = ArithmeticCoder(coder_settings)
        compressed_data.data = codr.encode(syms, predictor)
        return compressed_data
    
    def decompress(self, compressed_model):
        if not isinstance(compressed_model, CompressedModel):
            raise ValueError("Data should be in form of CompressedModel")
        if compressed_model.version != 1:
            raise ValueError("Version not supported")
        if compressed_model.vocab_code != byte_preprocessor().code:
            raise ValueError("Preprocessor not supported")
        data = compressed_model.data
        prpr = byte_preprocessor()
        dictionary = prpr.construct_dictionary_from_header(compressed_model.preprocessor_header)
        keras_model = SimpleGRUModel(dictionary.get_size())
        predictor = TfPredictionModel(dictionary, keras_model)
        coder_settings = ArithmeticCoderSettings()
        codr = ArithmeticCoder(coder_settings)
        syms = codr.decode(data, dictionary, predictor)
        return prpr.convert_from_symbols(syms)
    
class byte_codec_deep(base_codec):
    def __init__(self):
        enable_determinism(TF_SEED)
    
    def compress(self, data):
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        compressed_data = CompressedModel()
        compressed_data.version = 1
        prpr = byte_preprocessor()
        compressed_data.vocab_code = prpr.code
        syms, dictionary = prpr.convert_to_symbols(data)
        compressed_data.preprocessor_header = prpr.encode_dictionary_for_header(dictionary)
        keras_model = SimpleGRUModel(dictionary.get_size())
        predictor = TfPredictionModel(dictionary, keras_model)
        coder_settings = ArithmeticCoderSettings()
        codr = ArithmeticCoderDeep(coder_settings)
        compressed_data.data = codr.encode(syms, predictor)
        return compressed_data
    
    def decompress(self, compressed_model):
        if not isinstance(compressed_model, CompressedModel):
            raise ValueError("Data should be in form of CompressedModel")
        if compressed_model.version != 1:
            raise ValueError("Version not supported")
        if compressed_model.vocab_code != byte_preprocessor().code:
            raise ValueError("Preprocessor not supported")
        data = compressed_model.data
        prpr = byte_preprocessor()
        dictionary = prpr.construct_dictionary_from_header(compressed_model.preprocessor_header)
        keras_model = SimpleGRUModel(dictionary.get_size())
        predictor = TfPredictionModel(dictionary, keras_model)
        coder_settings = ArithmeticCoderSettings()
        codr = ArithmeticCoderDeep(coder_settings)
        syms = codr.decode(data, dictionary, predictor)
        return prpr.convert_from_symbols(syms)