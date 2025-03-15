#coder.py

import abc
import numpy as np
import os
import tempfile
import struct

class coder_base(abc.ABC):
    def __init__(self):
        """
        Initialize the arithmetic coder.
        """
        pass

    @abc.abstractmethod
    def encode(self, symbols, prediction_model):
        """
        Encodes a sequence of symbols into a compressed byte stream
        
        Args:
            symbols (list[Symbol]): The list of symbols to be encoded.
            prediction_model(base_prediction_model): The prediction model used to obtain probability distributions and for training.
        
        Returns:
            list[byte]: The encoded data a list of bytes.
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

class arithmetic_coder_settings:
    def __init__(self, scaling_factor=10000000, offset=1):
        self.scaling_factor = scaling_factor
        self.offset = offset

class arithmetic_coder(coder_base):
    def __init__(self, artihmetic_coder_settings):
        
        if not isinstance(artihmetic_coder_settings, arithmetic_coder_settings):
            raise ValueError("artihmetic_coder_settings must be an instance of arithmetic_coder_settings")
        
        self.scaling_factor = artihmetic_coder_settings.scaling_factor
        self.offset = artihmetic_coder_settings.offset
        
        self.low = 0
        self.high = (1 << 32) - 1
        self.state_bits = 32
        self.encoded_bits = [] 

    def encode(self, symbols, prediction_model):
        
        if symbols is None or not isinstance(symbols, list) or len(symbols) == 0:
            raise ValueError("symbols must be a non-empty list of Symbol objects")
        
        num_symbols = len(symbols)
        header = num_symbols.to_bytes(4, byteorder='big')

        low = self.low
        high = self.high
        output_bits = []
        context = []
        symbols_encoded = 0

        for symbol in symbols:
            bits_before = len(output_bits)
            
            probs = prediction_model.predict(context)
            cum_freq = self.probabilities_to_code(probs)
            total = cum_freq[-1]

            sorted_symbols = sorted(prediction_model.dictionary.symbols, key=lambda s: s.data)
            symbol_index = None
            for idx, s in enumerate(sorted_symbols):
                if s == symbol:
                    symbol_index = idx
                    break
            if symbol_index is None:
                raise ValueError("Symbol not found in dictionary")

            sym_low = cum_freq[symbol_index - 1] if symbol_index > 0 else 0
            sym_high = cum_freq[symbol_index]

            current_range = high - low + 1
            new_low = low + (sym_low * current_range) // total
            new_high = low + (sym_high * current_range) // total - 1
            low, high = new_low, new_high

            while (low >> (self.state_bits - 1)) == (high >> (self.state_bits - 1)):
                bit = low >> (self.state_bits - 1)
                output_bits.append(bit)
                low = (low << 1) & ((1 << self.state_bits) - 1)
                high = ((high << 1) & ((1 << self.state_bits) - 1)) | 1
            
            prediction_model.train(context, symbol)

            context.append(symbol)
            symbols_encoded += 1
            
            bits_after = len(output_bits)
            
        for _ in range(self.state_bits):
            bit = low >> (self.state_bits - 1)
            output_bits.append(bit)
            low = (low << 1) & ((1 << self.state_bits) - 1)

        byte_array = []
        for i in range(0, len(output_bits), 8):
            byte = 0
            for j in range(8):
                byte = byte << 1
                if i + j < len(output_bits):
                    byte |= output_bits[i + j]
            byte_array.append(byte)

        encoded_bytes = header + bytes(byte_array)
        return encoded_bytes

    def decode(self, data, dictionary, prediction_model):
        
        if not isinstance(data, bytes) or len(data) < 4:
            raise ValueError("data must be a byte string with at least 4 bytes for the header")
        if dictionary is None:
            raise ValueError("dictionary cannot be None")

        header = data[:4]
        num_symbols = int.from_bytes(header, byteorder='big')

        bitstream_data = data[4:]
        bitstream = []
        for byte in bitstream_data:
            for i in range(7, -1, -1):
                bitstream.append((byte >> i) & 1)

        low = self.low
        high = self.high
        code = 0
        bit_index = 0
        for i in range(self.state_bits):
            code = (code << 1) | (bitstream[bit_index] if bit_index < len(bitstream) else 0)
            bit_index += 1

        decoded_symbols = []
        context = []
        for symbol_count in range(num_symbols):
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

            sorted_symbols = sorted(dictionary.symbols, key=lambda s: s.data)
            decoded_symbol = sorted_symbols[symbol_index]
            
            decoded_symbols.append(decoded_symbol)

            prediction_model.train(context, decoded_symbol)

            sym_low = cum_freq[symbol_index - 1] if symbol_index > 0 else 0
            sym_high = cum_freq[symbol_index]
            new_low = low + (sym_low * current_range) // total
            new_high = low + (sym_high * current_range) // total - 1
            low, high = new_low, new_high

            while (low >> (self.state_bits - 1)) == (high >> (self.state_bits - 1)):
                low = (low << 1) & ((1 << self.state_bits) - 1)
                high = ((high << 1) & ((1 << self.state_bits) - 1)) | 1
                if bit_index < len(bitstream):
                    next_bit = bitstream[bit_index]
                else:
                    next_bit = 0
                bit_index += 1
                code = ((code << 1) & ((1 << self.state_bits) - 1)) | next_bit

            context.append(decoded_symbol)

        return decoded_symbols

    def probabilities_to_code(self, probs):
        """
        Convert a probability distribution into a cumulative frequency table.
        
        Args:
            probs (list or np.ndarray): Either a list/array of numbers (floats) or a list of SymbolFrequency objects.
        
        Returns:
            np.ndarray: A 1D numpy array of cumulative frequencies.
        
        Raises:
            ValueError: If the input probabilities do not sum to 1.
        """
        if isinstance(probs, list) and len(probs) > 0 and hasattr(probs[0], 'frequency'):
            probs_array = np.array([float(sf.frequency) for sf in probs], dtype=np.float64)
        else:
            probs_array = np.array(probs, dtype=np.float64)
        
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

class arithmetic_coder_deep:
    def __init__(self, coder_settings):
        """
        Initializes arithmetic_coder_deep with optional settings.
        """
        if not isinstance(coder_settings, arithmetic_coder_settings):
            raise ValueError("coder_settings must be an instance of arithmetic_coder_settings")
        
        self.scaling_factor = coder_settings.scaling_factor
        self.offset = coder_settings.offset

    def probabilities_to_code(self, probs):
        """
        Convert a probability distribution into a cumulative frequency table.
        
        Args:
            probs (list or np.ndarray): Either a list/array of numbers (floats) or a list of SymbolFrequency objects.
        
        Returns:
            np.ndarray: A 1D numpy array of cumulative frequencies.
        
        Raises:
            ValueError: If the probabilities do not sum to 1.
        """
        if isinstance(probs, list) and len(probs) > 0 and hasattr(probs[0], 'frequency'):
            probs_array = np.array([float(sf.frequency) for sf in probs], dtype=np.float64)
        else:
            probs_array = np.array(probs, dtype=np.float64)
        
        if not np.isclose(np.sum(probs_array), 1.0, atol=1e-5):
            raise ValueError("The input probabilities must sum to 1")
        
        freqs = np.round(probs_array * self.scaling_factor + self.offset).astype(np.int64)
        return np.cumsum(freqs)

    def code_to_probabilities(self, cum_freq):
        """
        Converts a cumulative frequency array into probabilities.
        
        Args:
            cum_freq (np.ndarray): 1-D array of cumulative frequencies.
            
        Returns:
            np.ndarray: Array of probabilities.
        """
        cum_freq = np.array(cum_freq)
        if cum_freq.ndim != 1:
            raise ValueError("cumulative frequency table must be one-dimensional")
        freqs = np.empty_like(cum_freq)
        freqs[0] = cum_freq[0]
        freqs[1:] = np.diff(cum_freq)
        total = cum_freq[-1]
        if total == 0:
            raise ValueError("Total frequency cannot be zero")
        probabilities = freqs / total
        return probabilities

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
        
        message_length = len(input_symbols)
        arithmetic_bitstream = self.arithmetic_encode_symbols(input_symbols, prediction_model, prediction_model.dictionary)
       
        msg_length_header = struct.pack(">I", message_length)
        arithmetic_bitstream = msg_length_header + arithmetic_bitstream
        
        return weights_header + weights_data + arithmetic_bitstream

    def arithmetic_encode_symbols(self, input_symbols, prediction_model, dictionary):
        """
        Performs arithmetic encoding on the symbol sequence.
        
        For each symbol (using its context), obtain the fixed probability distribution from
        prediction_model.predict(context), check for near-zero probabilities, convert to a cumulative frequency
        table, and update the [low, high] interval. Renormalization is applied to extract bits.
        
        Args:
            input_symbols (list): List of symbol objects.
            prediction_model: A prediction model instance.
            dictionary: A dictionary instance with symbol ordering.
        
        Returns:
            bytes: The arithmetic-coded bitstream (excluding the 4-byte message length header).
        """
        if not isinstance(input_symbols, list):
            raise ValueError("input_symbols must be a list.")
        if prediction_model is None or dictionary is None:
            raise ValueError("prediction_model and dictionary cannot be None.")

        low = 0.0
        high = 1.0
        underflow_count = 0
        output_bits = []  
        
        def output_bit(bit):
            output_bits.append(bit)
        
        epsilon = 1e-10
        
        for i, symbol in enumerate(input_symbols):
            context = input_symbols[:i]
            probs_sf = prediction_model.predict(context)
            if not isinstance(probs_sf, list):
                raise ValueError("Prediction output must be a list of SymbolFrequency objects.")
            try:
                float_probs = [float(sf.frequency) for sf in probs_sf]
            except Exception as e:
                raise ValueError("Failed to extract probability values from prediction output: " + str(e))
            if any(p < epsilon for p in float_probs):
                raise ValueError("Symbol probability is zero or too close to zero.")
            freqs = np.round(np.array(float_probs) * self.scaling_factor + self.offset).astype(np.int64)
            cumulative = np.cumsum(freqs)
            total = cumulative[-1]
            
            try:
                symbol_index = dictionary.get_index(symbol)
            except AttributeError:
                sorted_symbols = sorted(dictionary.symbols, key=lambda s: s.data)
                mapping = {s.data: idx for idx, s in enumerate(sorted_symbols)}
                if symbol.data not in mapping:
                    raise ValueError("Symbol not found in dictionary.")
                symbol_index = mapping[symbol.data]
            
            lower_freq = 0 if symbol_index == 0 else cumulative[symbol_index - 1]
            upper_freq = cumulative[symbol_index]
            
            range_width = high - low
            new_low = low + range_width * (lower_freq / total)
            new_high = low + range_width * (upper_freq / total)
            low, high = new_low, new_high
            
            while True:
                if high < 0.5:
                    output_bit(0)
                    for _ in range(underflow_count):
                        output_bit(1)
                    underflow_count = 0
                    low *= 2
                    high *= 2
                elif low >= 0.5:
                    output_bit(1)
                    for _ in range(underflow_count):
                        output_bit(0)
                    underflow_count = 0
                    low = (low - 0.5) * 2
                    high = (high - 0.5) * 2
                elif low >= 0.25 and high < 0.75:
                    underflow_count += 1
                    low = (low - 0.25) * 2
                    high = (high - 0.25) * 2
                else:
                    break
        
        underflow_count += 1
        if low < 0.25:
            output_bit(0)
            for _ in range(underflow_count):
                output_bit(1)
        else:
            output_bit(1)
            for _ in range(underflow_count):
                output_bit(0)
        
        bit_string = ''.join(str(b) for b in output_bits)
        pad_len = (8 - len(bit_string) % 8) % 8
        bit_string += '0' * pad_len
        output_bytes = bytearray()
        for i in range(0, len(bit_string), 8):
            byte = int(bit_string[i:i+8], 2)
            output_bytes.append(byte)
        return bytes(output_bytes)

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
        
        bit_string = ''.join(format(b, '08b') for b in bitstream_bytes)
        bit_index = 0
        def read_bit():
            nonlocal bit_index
            if bit_index >= len(bit_string):
                return 0
            b = int(bit_string[bit_index])
            bit_index += 1
            return b
        
        low = 0.0
        high = 1.0
        num_init_bits = 32
        code_value = 0.0
        for _ in range(num_init_bits):
            code_value = code_value * 2 + read_bit()
        code_value /= 2**num_init_bits
        
        decoded_symbols = []
        epsilon = 1e-10
        
        for _ in range(message_length):
            context = decoded_symbols[:] 
            probs_sf = prediction_model.predict(context)
            if not isinstance(probs_sf, list):
                raise ValueError("Prediction output must be a list of SymbolFrequency objects.")
            try:
                float_probs = [float(sf.frequency) for sf in probs_sf]
            except Exception as e:
                raise ValueError("Failed to extract probability values during decoding: " + str(e))
            if any(p < epsilon for p in float_probs):
                raise ValueError("Symbol probability is zero or too close to zero during decoding.")
            freqs = np.round(np.array(float_probs) * self.scaling_factor + self.offset).astype(np.int64)
            cumulative = np.cumsum(freqs)
            total = cumulative[-1]
            
            range_width = high - low
            scaled_value = (code_value - low) / range_width
            target = scaled_value * total
            symbol_index = 0
            while symbol_index < len(cumulative) and cumulative[symbol_index] <= target:
                symbol_index += 1
            try:
                symbol = dictionary.get_symbol_by_index(symbol_index)
            except AttributeError:
                sorted_symbols = sorted(dictionary.symbols, key=lambda s: s.data)
                if symbol_index >= len(sorted_symbols):
                    raise ValueError("Decoded symbol index out of range.")
                symbol = sorted_symbols[symbol_index]
            decoded_symbols.append(symbol)
            
            lower_freq = 0 if symbol_index == 0 else cumulative[symbol_index - 1]
            upper_freq = cumulative[symbol_index]
            new_low = low + range_width * (lower_freq / total)
            new_high = low + range_width * (upper_freq / total)
            low, high = new_low, new_high
            
            while True:
                if high < 0.5:
                    low *= 2
                    high *= 2
                    code_value = code_value * 2 + read_bit() / (2**num_init_bits)
                elif low >= 0.5:
                    low = (low - 0.5) * 2
                    high = (high - 0.5) * 2
                    code_value = (code_value - 0.5) * 2 + read_bit() / (2**num_init_bits)
                elif low >= 0.25 and high < 0.75:
                    low = (low - 0.25) * 2
                    high = (high - 0.25) * 2
                    code_value = (code_value - 0.25) * 2 + read_bit() / (2**num_init_bits)
                else:
                    break
        return decoded_symbols