# coder.py

import abc
import numpy as np
import os
import tempfile
import struct

from .models import Symbol
from .logger import *

class CoderBase(abc.ABC):
    def __init__(self):
        """Initialize the arithmetic coder."""
        pass

    @abc.abstractmethod
    def encode(self, symbols, prediction_model):
        """
        Encodes a sequence of symbols into a compressed byte stream.
        
        Args:
            symbols (list[Symbol]): The list of symbols to be encoded.
            prediction_model(base_prediction_model): The prediction model used to obtain probability distributions and for training.
        
        Returns:
            bytes: The encoded data.
        """
        pass

    @abc.abstractmethod
    def decode(self, data, dictionary, prediction_model):
        """
        Decodes a compressed byte stream back into a list of Symbol objects.
        
        Args:
            data (bytes): The encoded data.
            dictionary (Dictionary): The dictionary used during encoding.
            prediction_model: The prediction model used to assist in decoding and for training.
            
        Returns:
            list[Symbol]: The decoded list of symbols.
        """
        pass
    
    @abc.abstractmethod
    def get_coder_code(self):
        """
        Returns the code for the coder. 
        """
        pass

class ArithmeticCoderSettings:
    def __init__(self, scaling_factor=10000000, offset=1):
        self.scaling_factor = scaling_factor
        self.offset = offset

class ArithmeticCodec:
    """
    Shared arithmetic coding logic.
    This class consolidates common code such as probability conversion functions and 
    bitstream packing/unpacking functions, as well as the integer-based arithmetic encoding/decoding loops.
    """
    def __init__(self, settings: ArithmeticCoderSettings, logger=None):
        self.scaling_factor = settings.scaling_factor
        self.offset = settings.offset
        self.logger = logger

    def probabilities_to_code(self, probs):
        """
        Convert a probability distribution into a cumulative frequency table.
        
        Args:
            probs (list or np.ndarray): Either a list/array of numbers (floats) 
                                         or a list of objects with a 'frequency' attribute.
        
        Returns:
            np.ndarray: A 1D numpy array of cumulative frequencies.
            
        Raises:
            ValueError: If the input probabilities do not sum to 1.
        """
        if probs is None:
            raise ValueError("The input probabilities must not be None")
        if isinstance(probs, list) and len(probs) > 0 and hasattr(probs[0], 'frequency'):
            probs_array = np.array([float(sf.frequency) for sf in probs], dtype=np.float64)
        else:
            try:
                probs_array = np.array(probs, dtype=np.float64)
            except Exception as e:
                raise ValueError("Invalid probability values provided: " + str(e))
        
        if not np.isclose(np.sum(probs_array), 1.0, atol=1e-5):
            raise ValueError("The input probabilities must sum to 1")
        
        freqs = np.round(probs_array * self.scaling_factor + self.offset).astype(np.int64)
        return np.cumsum(freqs)

    def code_to_probabilities(self, cum_freq):
        """
        Convert a cumulative frequency table back into a probability distribution.
        
        Args:
            cum_freq (list or np.ndarray): A one-dimensional cumulative frequency table.
        
        Returns:
            np.ndarray: A probability distribution (sums to 1).
        
        Raises:
            ValueError: If cum_freq is not a one-dimensional array.
        """
        if cum_freq is None:
            raise ValueError("cumulative frequency table must not be None")
        cum_freq = np.array(cum_freq)
        if cum_freq.ndim != 1:
            raise ValueError("cumulative frequency table must be one-dimensional")
        freqs = np.empty_like(cum_freq)
        freqs[0] = cum_freq[0]
        freqs[1:] = cum_freq[1:] - cum_freq[:-1]
        total = cum_freq[-1]
        if total == 0:
            raise ValueError("Total frequency cannot be zero")
        probabilities = freqs / total
        return probabilities

    def pack_bits_to_bytes(self, bits):
        """
        Pack a list of bits (0s and 1s) into a byte stream.
        
        Args:
            bits (list[int]): List of bits.
        
        Returns:
            bytes: Packed bytes.
        """
        if bits is None:
            raise ValueError("bits cannot be None")
        byte_array = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = byte << 1
                if i + j < len(bits):
                    byte |= bits[i + j]
            byte_array.append(byte)
        return bytes(byte_array)

    def unpack_bytes_to_bits(self, data):
        """
        Unpack a byte stream into a list of bits (MSB first).
        
        Args:
            data (bytes): The byte stream.
        
        Returns:
            list[int]: List of bits.
        """
        if data is None:
            raise ValueError("data cannot be None")
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    def integer_arithmetic_encode_symbols(self, symbols, prediction_model, dictionary, state_bits=32, train_callback=None):
        """
        Shared integer-based arithmetic encoding.
        
        Args:
            symbols (list): List of symbol objects.
            prediction_model: Prediction model to provide probability distributions.
            dictionary: Dictionary for symbol ordering.
            state_bits (int): Number of bits for state representation.
            train_callback (callable): Function called after encoding each symbol.
                                       Should accept (context, symbol). If None, no training is done.
        
        Returns:
            bytes: The packed arithmetic-coded bitstream.
        """
        low = 0
        high = (1 << state_bits) - 1
        output_bits = []
        context = []

        for symbol in symbols:
            bits_before = len(output_bits)  # for logging
            probs = prediction_model.predict(context)
            
            
            cum_freq = self.probabilities_to_code(probs)
            total = cum_freq[-1]

            try:
                symbol_index = dictionary.get_index(symbol)
            except AttributeError:
                sorted_symbols = sorted(dictionary.symbols, key=lambda s: s.data)
                symbol_index = None
                for idx, s in enumerate(sorted_symbols):
                    if s == symbol:
                        symbol_index = idx
                        break
                if symbol_index is None:
                    raise ValueError("Symbol not found in dictionary")
                
            
            if self.logger is not None:
                self.logger.log(EncodedSymbolProbability(symbol, probs[symbol_index].frequency.item()))
            
            sym_low = cum_freq[symbol_index - 1] if symbol_index > 0 else 0
            sym_high = cum_freq[symbol_index]

            current_range = high - low + 1
            new_low = low + (sym_low * current_range) // total
            new_high = low + (sym_high * current_range) // total - 1
            low, high = new_low, new_high

            while (low >> (state_bits - 1)) == (high >> (state_bits - 1)):
                bit = low >> (state_bits - 1)
                output_bits.append(bit)
                low = (low << 1) & ((1 << state_bits) - 1)
                high = ((high << 1) & ((1 << state_bits) - 1)) | 1

            if train_callback is not None:
                train_callback(context, symbol)
            context.append(symbol)
            
            bits_after = len(output_bits)  # for logging
            encoded_bits = bits_after - bits_before # for logging
            symbol_size = len(symbol.data) * 8 # for logging
            if self.logger is not None:
                self.logger.log(CodingLog(symbol_size, encoded_bits))
                self.logger.log(CodingProgressStep(len(output_bits), len(symbols)))
                self.logger.log(PredictionModelTrainingProgressStep(len(context), len(symbols)))
            

        for _ in range(state_bits):
            bit = low >> (state_bits - 1)
            output_bits.append(bit)
            low = (low << 1) & ((1 << state_bits) - 1)

        return self.pack_bits_to_bytes(output_bits)

    def integer_arithmetic_decode(self, bitstream, num_symbols, prediction_model, dictionary, state_bits=32, train_callback=None):
        """
        Shared integer-based arithmetic decoding.
        
        Args:
            bitstream (list[int]): List of bits representing the encoded data.
            num_symbols (int): Number of symbols expected.
            prediction_model: Prediction model to provide probability distributions.
            dictionary: Dictionary for symbol ordering.
            state_bits (int): Number of bits for state representation.
            train_callback (callable): Function called after decoding each symbol.
                                       Should accept (context, symbol). If None, no training is done.
        
        Returns:
            list: The list of decoded symbols.
        """
        low = 0
        high = (1 << state_bits) - 1
        code = 0
        bit_index = 0
        for _ in range(state_bits):
            code = (code << 1) | (bitstream[bit_index] if bit_index < len(bitstream) else 0)
            bit_index += 1

        decoded_symbols = []
        context = []
        for _ in range(num_symbols):
            probs = prediction_model.predict(context)
            
            cum_freq = self.probabilities_to_code(probs)
            total = cum_freq[-1]
            current_range = high - low + 1
            scaled_value = ((code - low + 1) * total - 1) // current_range

            symbol_index = None
            for i, freq in enumerate(cum_freq):
                if scaled_value < freq:
                    symbol_index = i
                    break
            if symbol_index is None:
                raise ValueError("Failed to decode symbol: no matching frequency range found.")

            try:
                decoded_symbol = dictionary.get_symbol_by_index(symbol_index)
            except AttributeError:
                sorted_symbols = sorted(dictionary.symbols, key=lambda s: s.data)
                if symbol_index >= len(sorted_symbols):
                    raise ValueError("Decoded symbol index out of range.")
                decoded_symbol = sorted_symbols[symbol_index]
            
            if train_callback is not None:
                train_callback(context, decoded_symbol)
            decoded_symbols.append(decoded_symbol)
            context.append(decoded_symbol)

            sym_low = cum_freq[symbol_index - 1] if symbol_index > 0 else 0
            sym_high = cum_freq[symbol_index]
            new_low = low + (sym_low * current_range) // total
            new_high = low + (sym_high * current_range) // total - 1
            low, high = new_low, new_high

            while (low >> (state_bits - 1)) == (high >> (state_bits - 1)):
                low = (low << 1) & ((1 << state_bits) - 1)
                high = ((high << 1) & ((1 << state_bits) - 1)) | 1
                next_bit = bitstream[bit_index] if bit_index < len(bitstream) else 0
                bit_index += 1
                code = ((code << 1) & ((1 << state_bits) - 1)) | next_bit

        return decoded_symbols

class ArithmeticCoder(CoderBase):
    def __init__(self, arithmetic_coder_settings, logger=None):
        if not isinstance(arithmetic_coder_settings, ArithmeticCoderSettings):
            raise ValueError("arithmetic_coder_settings must be an instance of ArithmeticCoderSettings")
        
        self.state_bits = 32
        self.codec = ArithmeticCodec(arithmetic_coder_settings, logger)
        
        self.coder_code = 1
        
        self.logger = logger

    def encode(self, symbols, prediction_model):
        if symbols is None or not isinstance(symbols, list) or len(symbols) == 0:
            raise ValueError("symbols must be a non-empty list of Symbol objects")
        num_symbols = len(symbols)
        header = num_symbols.to_bytes(4, byteorder='big')
        bitstream_bytes = self.codec.integer_arithmetic_encode_symbols(
            symbols, 
            prediction_model, 
            prediction_model.dictionary, 
            self.state_bits, 
            train_callback=lambda ctx, sym: prediction_model.train(ctx, sym)
        )
        return header + bitstream_bytes

    def decode(self, data, dictionary, prediction_model):
        if not isinstance(data, bytes) or len(data) < 4:
            raise ValueError("data must be a byte string with at least 4 bytes for the header")
        if dictionary is None:
            raise ValueError("dictionary cannot be None")
        header = data[:4]
        num_symbols = int.from_bytes(header, byteorder='big')
        bitstream_data = data[4:]
        bitstream = self.codec.unpack_bytes_to_bits(bitstream_data)
        decoded_symbols = self.codec.integer_arithmetic_decode(
            bitstream, 
            num_symbols, 
            prediction_model, 
            dictionary, 
            self.state_bits, 
            train_callback=lambda ctx, sym: prediction_model.train(ctx, sym)
        )
        return decoded_symbols
    
    def get_coder_code(self):
        return self.coder_code

class ArithmeticCoderDeep(CoderBase):
    def __init__(self, coder_settings, logger=None):
        if not isinstance(coder_settings, ArithmeticCoderSettings):
            raise ValueError("coder_settings must be an instance of ArithmeticCoderSettings")
        self.state_bits = 32
        self.codec = ArithmeticCodec(coder_settings, logger)
        self.logger = logger
        
        self.coder_code = 2

    def encode(self, input_symbols, prediction_model):
        if not isinstance(input_symbols, list):
            raise ValueError("input_symbols must be a list.")
        if prediction_model is None:
            raise ValueError("prediction_model cannot be None.")
        if not hasattr(prediction_model, "train") or not hasattr(prediction_model, "save_model"):
            raise ValueError("prediction_model must have train and save_model methods.")
        if not hasattr(prediction_model, "dictionary"):
            raise ValueError("prediction_model must have a dictionary attribute.")

        for i, symbol in enumerate(input_symbols):
            context = input_symbols[:i]
            prediction_model.train(context, symbol)
            if self.logger is not None:
                self.logger.log(PredictionModelTrainingProgressStep(i, len(input_symbols)))
        
        temp_dir = tempfile.gettempdir()
        temp_filename = next(tempfile._get_candidate_names()) + ".weights.h5"
        temp_filepath = os.path.join(temp_dir, temp_filename)
        prediction_model.save_model(temp_filepath)
        if not os.path.exists(temp_filepath):
            raise IOError("Model weights file was not created.")
        with open(temp_filepath, "rb") as f:
            weights_data = f.read()
        os.remove(temp_filepath)
        weights_header = struct.pack(">Q", len(weights_data))
        
        msg_length_header = struct.pack(">I", len(input_symbols))
        arithmetic_bitstream = self.codec.integer_arithmetic_encode_symbols(
            input_symbols, 
            prediction_model, 
            prediction_model.dictionary, 
            self.state_bits, 
            train_callback=None
        )
        arithmetic_bitstream = msg_length_header + arithmetic_bitstream
        
        return weights_header + weights_data + arithmetic_bitstream

    def decode(self, encoded_data, dictionary, prediction_model):
        if not isinstance(encoded_data, bytes):
            raise ValueError("encoded_data must be a bytes object.")
        if len(encoded_data) < 8:
            raise ValueError("Encoded data is too short.")
        if dictionary is None or prediction_model is None:
            raise ValueError("dictionary and prediction_model cannot be None.")
        
        weights_header = encoded_data[:8]
        weights_length = struct.unpack(">Q", weights_header)[0]
        if len(encoded_data) < 8 + weights_length:
            raise ValueError("Encoded data is missing weights.")
        weights_data = encoded_data[8:8+weights_length]
        arithmetic_data = encoded_data[8+weights_length:]
        
        temp_dir = tempfile.gettempdir()
        temp_filename = next(tempfile._get_candidate_names()) + ".weights.h5"
        temp_filepath = os.path.join(temp_dir, temp_filename)
        with open(temp_filepath, "wb") as f:
            f.write(weights_data)
        prediction_model.load_model(temp_filepath)
        os.remove(temp_filepath)
        
        if len(arithmetic_data) < 4:
            raise ValueError("Arithmetic data missing message length header.")
        msg_length_header = arithmetic_data[:4]
        message_length = struct.unpack(">I", msg_length_header)[0]
        bitstream_bytes = arithmetic_data[4:]
        bitstream = self.codec.unpack_bytes_to_bits(bitstream_bytes)
        decoded_symbols = self.codec.integer_arithmetic_decode(
            bitstream, 
            message_length, 
            prediction_model, 
            dictionary, 
            self.state_bits, 
            train_callback=None
        )
        return decoded_symbols
    
    def get_coder_code(self):
        self.coder_code = 2
        
        
def get_coder(code, logger=None):
    if code == 1:
        return ArithmeticCoder(ArithmeticCoderSettings(), logger)
    elif code == 2:
        return ArithmeticCoderDeep(ArithmeticCoderSettings(), logger=logger)
    else:
        raise ValueError("Unknown coder code: " + str(code))
