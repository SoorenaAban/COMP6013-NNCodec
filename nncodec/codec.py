#codec.py

import abc

from .prediction_models import tf_prediction_model
from .preprocessors import byte_preprocessor
from .coder import arithmetic_coder, arithmetic_coder_deep
from .models import CompressedModel

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
class byte_codec(base_codec):
    def compress(self, data):
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        compressed_data = CompressedModel()
        compressed_data.version = 1
        prpr = byte_preprocessor()
        compressed_data.vocab_code = prpr.code
        syms, dictionary = prpr.convert_to_symbols(data)
        compressed_data.preprocessor_header = prpr.encode_dictionary_for_header(dictionary)
        predictor = tf_prediction_model(dictionary)
        codr = arithmetic_coder()
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
        predictor = tf_prediction_model(dictionary)
        codr = arithmetic_coder()
        syms = codr.decode(data, dictionary, predictor)
        return prpr.convert_from_symbols(syms)
    
class byte_codec_deep(base_codec):
    def compress(self, data):
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        compressed_data = CompressedModel()
        compressed_data.version = 1
        prpr = byte_preprocessor()
        compressed_data.vocab_code = prpr.code
        syms, dictionary = prpr.convert_to_symbols(data)
        compressed_data.preprocessor_header = prpr.encode_dictionary_for_header(dictionary)
        predictor = tf_prediction_model(dictionary)
        codr = arithmetic_coder_deep()
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
        predictor = tf_prediction_model(dictionary)
        codr = arithmetic_coder_deep()
        syms = codr.decode(data, dictionary, predictor)
        return prpr.convert_from_symbols(syms)