#coder.py

import abc
from .models import Symbol
import numpy as np

class coder_base(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def encode(self, symbols, dictionary, preprocessor, prediction_model):
        """"returns the encoded data to be written to the file in form of bytes
        symbols: list of symbols to be encoded
        dictionary: the dictionary of symbols
        preprocessor: the preprocessor used to preprocess the data
        prediction_model: the prediction model used     to predict the symbols

        returns the encoded data in form of bytes
        """
        pass

    @abc.abstractmethod
    def decode(self, data, dictionary, preprocessor, prediction_model):
        """returns symbols that represent the decoded data"""
        pass

class arithmetic_coder(coder_base):
    def __init__(self, settings_override=None):
        """
        Initialize the arithmetic coder.
        
        Tunable parameters are read from the settings file by default or can be injected via
        the settings_override dictionary. Expected keys:
            - 'ARITH_SCALING_FACTOR'
            - 'ARITH_OFFSET'
        
        Args:
            settings_override (dict, optional): Dictionary with settings overrides.
        """
        super().__init__()
        if settings_override is None:
            try:
                import settings
                self.scaling_factor = getattr(settings, 'ARITH_SCALING_FACTOR', 10000000)
                self.offset = getattr(settings, 'ARITH_OFFSET', 1)
            except ImportError:
                self.scaling_factor = 10000000
                self.offset = 1
        else:
            self.scaling_factor = settings_override.get('ARITH_SCALING_FACTOR', 10000000)
            self.offset = settings_override.get('ARITH_OFFSET', 1)
        
        # Initialize internal state for arithmetic coding.
        self.low = 0
        self.high = (1 << 32) - 1
        self.state_bits = 32
        self.encoded_bits = []  # Internal buffer for output bits.

    def encode(self, symbols, dictionary, preprocessor, prediction_model):
        """
        Encodes a sequence of symbols into a compressed byte stream using arithmetic coding.
        
        Args:
            symbols (list): List of Symbol objects to be encoded.
            dictionary (Dictionary): The dictionary of symbols.
            preprocessor: The preprocessor used to preprocess the data.
            prediction_model: The prediction model used to predict the next symbol.
        
        Returns:
            bytes: The encoded data as a byte string.
        """
        # Placeholder implementation – to be replaced with full arithmetic coding logic.
        if symbols is None or not isinstance(symbols, list) or len(symbols) == 0:
            raise ValueError("symbols must be a non-empty list of Symbol objects")
        if dictionary is None:
            raise ValueError("dictionary cannot be None")
        # For now, return a fixed byte string.
        return b"encoded_data"
    
    def decode(self, data, dictionary, preprocessor, prediction_model):
        """
        Decodes a compressed byte stream back into the original symbols using arithmetic decoding.
        
        Args:
            data (bytes): The encoded data.
            dictionary (Dictionary): The dictionary used during encoding.
            preprocessor: The preprocessor used during encoding.
            prediction_model: The prediction model used to assist in decoding.
        
        Returns:
            list: A list of Symbol objects representing the decoded data.
        """
        if not isinstance(data, bytes) or len(data) == 0:
            raise ValueError("data must be a non-empty byte string")
        if dictionary is None:
            raise ValueError("dictionary cannot be None")
        # Placeholder implementation – to be replaced with full arithmetic decoding logic.
        return [Symbol(b'a'), Symbol(b'b')]
    
    def probabilities_to_code(self, probabilities):
        """
        Converts a probability distribution into its cumulative frequency representation
        for arithmetic coding.
        
        This function scales the input probabilities by a tunable factor and adds an offset,
        then computes the cumulative frequency table. The tunable parameters are:
            - scaling_factor: Scales each probability.
            - offset: Added to each scaled probability to ensure nonzero frequencies.
        
        Args:
            probabilities (list or np.ndarray): A probability distribution over symbols.
                Expected to sum to 1.
        
        Returns:
            np.ndarray: A 1D numpy array containing the cumulative frequency table (integers).
        
        Raises:
            ValueError: If probabilities is not a list/array or if they do not sum to 1.
        """
        # Validate input type.
        if probabilities is None or not isinstance(probabilities, (list, np.ndarray)):
            raise ValueError("probabilities must be a list or numpy array")
        probs = np.array(probabilities, dtype=np.float64)
        # Validate that probabilities sum to 1 (within a tolerance).
        if not np.isclose(np.sum(probs), 1.0, atol=1e-5):
            raise ValueError("The input probabilities must sum to 1")
        
        # Scale probabilities and add offset.
        # For each probability p, compute: freq = round(p * scaling_factor + offset)
        freqs = np.round(probs * self.scaling_factor + self.offset).astype(np.int64)
        
        # Compute the cumulative frequency table.
        cum_freq = np.cumsum(freqs)
        return cum_freq

    def code_to_probabilities(self, cum_freq):
        """
        Converts a cumulative frequency table back into a probability distribution.
        
        Given a cumulative frequency table (1D numpy array) computed by probabilities_to_code,
        this function calculates the frequency differences and normalizes them to obtain the original
        probability distribution.
        
        Args:
            cum_freq (list or np.ndarray): A cumulative frequency table (1D array of integers).
        
        Returns:
            np.ndarray: A 1D numpy array of probabilities that sum to 1.
        
        Raises:
            ValueError: If cum_freq is not a one-dimensional array.
        """
        # Validate input type.
        cum_freq = np.array(cum_freq)
        if cum_freq.ndim != 1:
            raise ValueError("cumulative frequency table must be a one-dimensional array")
        
        # Compute individual frequencies.
        freqs = np.empty_like(cum_freq)
        freqs[0] = cum_freq[0]
        freqs[1:] = cum_freq[1:] - cum_freq[:-1]
        total = cum_freq[-1]
        if total == 0:
            raise ValueError("Total frequency cannot be zero")
        probabilities = freqs / total
        return probabilities

    @property
    def current_state(self):
        """
        Return the current internal state of the arithmetic coder.
        
        Returns:
            dict: Contains keys 'low', 'high', and 'state_bits'.
        """
        return {"low": self.low, "high": self.high, "state_bits": self.state_bits}