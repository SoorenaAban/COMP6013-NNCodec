#coder.py

import abc
import numpy as np

from .models import Symbol

class coder_base(abc.ABC):
    def __init__(self, settings_override=None):
        """
        Initialize the arithmetic coder.
        (Todo: add tuning parameters later)
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

class arithmetic_coder(coder_base):
    def __init__(self, settings_override=None):
        
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
        
        self.low = 0
        self.high = (1 << 32) - 1
        self.state_bits = 32
        self.encoded_bits = [] 

    def encode(self, symbols, prediction_model):
        
        if symbols is None or not isinstance(symbols, list) or len(symbols) == 0:
            raise ValueError("symbols must be a non-empty list of Symbol objects")
        
        # Add header: store number of symbols (4 bytes, big-endian)
        num_symbols = len(symbols)
        header = num_symbols.to_bytes(4, byteorder='big')

        low = self.low
        high = self.high
        output_bits = []
        context = []
        symbols_encoded = 0

        for symbol in symbols:
            probs = prediction_model.predict(context)
            cum_freq = self.probabilities_to_code(probs)
            total = cum_freq[-1]

            # Determine the symbol's index using the sorted dictionary.
            sorted_symbols = sorted(prediction_model.dictionary.symbols, key=lambda s: s.data)
            symbol_index = None
            for idx, s in enumerate(sorted_symbols):
                if s == symbol:
                    symbol_index = idx
                    break
            if symbol_index is None:
                raise ValueError("Symbol not found in dictionary")

            # Retrieve the frequency range for this symbol.
            sym_low = cum_freq[symbol_index - 1] if symbol_index > 0 else 0
            sym_high = cum_freq[symbol_index]

            current_range = high - low + 1
            new_low = low + (sym_low * current_range) // total
            new_high = low + (sym_high * current_range) // total - 1
            low, high = new_low, new_high

            # Output bits while the most significant bits of low and high agree.
            while (low >> (self.state_bits - 1)) == (high >> (self.state_bits - 1)):
                bit = low >> (self.state_bits - 1)
                output_bits.append(bit)
                low = (low << 1) & ((1 << self.state_bits) - 1)
                high = ((high << 1) & ((1 << self.state_bits) - 1)) | 1
            
            # **Deterministic Training Update:**  
            # Train the prediction model with the current context and the actual symbol.
            prediction_model.train(context, symbol)

            # Append the symbol to the context.
            context.append(symbol)
            symbols_encoded += 1
            print(f"[INFO] Encoded symbol: '{symbol.data}'. Number of encoded symbols: {symbols_encoded}")

        # Flush the remaining bits: output state_bits bits.
        for _ in range(self.state_bits):
            bit = low >> (self.state_bits - 1)
            output_bits.append(bit)
            low = (low << 1) & ((1 << self.state_bits) - 1)

        # Pack bits into bytes.
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
            print(f"[INFO] Decoded symbol: '{decoded_symbol.data}'. Number of decoded symbols: {len(decoded_symbols) + 1}")
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

    def probabilities_to_code(self, probabilities):
        """
        Convert a probability distribution into a cumulative frequency table.
        
        Args:
            probabilities (list or np.ndarray): Either a list/array of floats (probabilities)
                or a list of SymbolFrequency objects.
        
        Returns:
            np.ndarray: A 1D numpy array of cumulative frequencies.
        
        Raises:
            ValueError: If the input is not of an expected type or if the probabilities do not sum to 1.
        """
        if isinstance(probabilities, list) and len(probabilities) > 0 and hasattr(probabilities[0], 'frequency'):
            probs_array = np.array([sf.frequency for sf in probabilities], dtype=np.float64)
        else:
            probs_array = np.array(probabilities, dtype=np.float64)
        
        if not np.isclose(np.sum(probs_array), 1.0, atol=1e-5):
            raise ValueError("The input probabilities must sum to 1")
        
        freqs = np.round(probs_array * self.scaling_factor + self.offset).astype(np.int64)
        
        cum_freq = np.cumsum(freqs)
        
        return cum_freq

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

    # @property
    # def current_state(self):
    #     """
    #     Return the current internal state of the arithmetic coder.
        
    #     Returns:
    #         dict: Contains 'low', 'high', and 'state_bits'.
    #     """
    #     return {"low": self.low, "high": self.high, "state_bits": self.state_bits}
