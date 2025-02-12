#codec.py

import abc

from .prediction_models import tf_prediction_model
from .preprocessors import byte_preprocessor
from .coder import arithmetic_coder

class base_codec(abc.ABC):
    """The base class for the codecs."""
    @abc.abstractmethod
    def compress(self, data):
        """
        Compresses the input data.

        Args:
            data (list[bytes]): The list of bytes representing the data
            
        Returns:
            list[bytes]: The list of bytes representing the encoded data
        """
        pass
    @abc.abstractmethod
    def decompress(self, data):
        """
        Decompresses encoded data.

        Args:
            data (list[bytes]): The list of bytes representing the encoded data
            
        Returns:
            list[bytes]: The list of bytes representing the decoded data
        """
        pass
class byte_codec(base_codec):
    def compress(self, data):
        prpr = byte_preprocessor()
        syms, dictionary = prpr.convert_to_symbols(data)
        predictor = tf_prediction_model(dictionary)
        codr = arithmetic_coder()
        return codr.encode(syms, predictor)
    
    def decompress(self, data):
        prpr = byte_preprocessor()
        dictionary = prpr.construct_dictionary_from_header("")
        predictor = tf_prediction_model(dictionary)
        codr = arithmetic_coder()
        syms = codr.decode(data, dictionary, predictor)
        return prpr.convert_from_symbols(syms)