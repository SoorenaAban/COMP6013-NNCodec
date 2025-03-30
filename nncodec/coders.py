"""
coders.py



"""


import abc
import numpy as np
import os
import tempfile
import struct
from io import BytesIO
from typing import Any, List, Optional, Callable, IO

from .logger import Logger, EncodedSymbolProbability, CodingLog, CodingProgressStep, PredictionModelTrainingProgressStep
from .models import Symbol, Dictionary
from .prediction_models import BasePredictionModel
from .validators import validate_type  


class CoderBase(abc.ABC):
    """
    Abstract base class for coders.
    """

    def __init__(self) -> None:
        """Initialize the coder."""
        pass

    @abc.abstractmethod
    def encode(self, symbols: List[Symbol], prediction_model: BasePredictionModel) -> bytes:
        """
        Encode a sequence of symbols into a compressed byte stream.
        
        Args:
            symbols (List[Symbol]): The list of symbols to be encoded.
            prediction_model (BasePredictionModel): The prediction model used to obtain probability distributions and for training.
        
        Returns:
            bytes: The encoded data.
        """
        pass

    @abc.abstractmethod
    def decode(self, data: bytes, dictionary: Dictionary, prediction_model: BasePredictionModel) -> List[Symbol]:
        """
        Decode a compressed byte stream back into a list of Symbol objects.
        
        Args:
            data (bytes): The encoded data.
            dictionary (Dictionary): The dictionary used during encoding.
            prediction_model (BasePredictionModel): The prediction model used to assist in decoding and for training.
            
        Returns:
            List[Symbol]: The decoded list of symbols.
        """
        pass

    @abc.abstractmethod
    def get_coder_code(self) -> int:
        """
        Get the code for the coder.
        
        Returns:
            int: The coder code.
        """
        pass


class ArithmeticCoderSettings:
    """
    Settings for the arithmetic coder.
    """

    def __init__(self, scaling_factor: int = 10000000, offset: int = 1) -> None:
        self.scaling_factor: int = scaling_factor
        self.offset: int = offset


class BitOutputStream:
    """
    A helper class to write bits to an underlying binary stream.
    """

    def __init__(self, out: IO[bytes]) -> None:
        """
        Initialize with an underlying output stream (e.g., a file opened in binary mode).
        
        Args:
            out (IO[bytes]): The output stream.
        """
        self.out: IO[bytes] = out
        self.current_byte: int = 0
        self.num_bits_filled: int = 0

    def write(self, bit: int) -> None:
        """
        Write a single bit (0 or 1) to the stream.
        
        Args:
            bit (int): The bit to write.
        
        Raises:
            ValueError: If the bit is not 0 or 1.
        """
        if bit not in (0, 1):
            raise ValueError("Bit must be 0 or 1")
        self.current_byte = (self.current_byte << 1) | bit
        self.num_bits_filled += 1
        if self.num_bits_filled == 8:
            self.flush_current_byte()

    def flush_current_byte(self) -> None:
        """
        Write the current byte to the underlying stream and reset the buffer.
        """
        self.out.write(bytes((self.current_byte,)))
        self.current_byte = 0
        self.num_bits_filled = 0

    def finish(self) -> None:
        """
        Flush any remaining bits to the stream by padding with zeros.
        """
        if self.num_bits_filled > 0:
            self.current_byte = self.current_byte << (8 - self.num_bits_filled)
            self.flush_current_byte()
        self.out.flush()

    def close(self) -> None:
        """
        Finish writing and close the underlying stream.
        """
        self.finish()
        self.out.close()


class BitInputStream:
    """
    A helper class to read bits from an underlying binary stream.
    """

    def __init__(self, inp: IO[bytes]) -> None:
        """
        Initialize with an underlying input stream (e.g., a file opened in binary mode).
        
        Args:
            inp (IO[bytes]): The input stream.
        """
        self.inp: IO[bytes] = inp
        self.current_byte: int = 0
        self.num_bits_remaining: int = 0

    def read(self) -> int:
        """
        Read a single bit from the stream.
        
        Returns:
            int: 0 or 1 for a valid bit, or -1 if no more bits are available.
        """
        if self.num_bits_remaining == 0:
            byte = self.inp.read(1)
            if len(byte) == 0:
                return -1
            self.current_byte = byte[0]
            self.num_bits_remaining = 8
        self.num_bits_remaining -= 1
        return (self.current_byte >> self.num_bits_remaining) & 1

    def close(self) -> None:
        """
        Close the underlying input stream.
        """
        self.inp.close()


class ArithmeticCodec:
    """ Shared arithmetic coding logic. """

    def __init__(self, settings: ArithmeticCoderSettings, logger: Optional[Logger] = None) -> None:
        self.scaling_factor: int = settings.scaling_factor
        self.offset: int = settings.offset
        self.logger: Optional[Logger] = logger

    def probabilities_to_code(self, probs: Any) -> np.ndarray:
        """
        Convert a probability distribution into a cumulative frequency table.
        
        Args:
            probs (Any): Either a list/array of numbers (floats) or a list of objects with a 'frequency' attribute.
        
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

    def code_to_probabilities(self, cum_freq: Any) -> np.ndarray:
        """
        Convert a cumulative frequency table back into a probability distribution.
        
        Args:
            cum_freq (Any): A one-dimensional cumulative frequency table.
        
        Returns:
            np.ndarray: A probability distribution that sums to 1.
        
        Raises:
            ValueError: If cum_freq is not a one-dimensional array or if total frequency is zero.
        """
        if cum_freq is None:
            raise ValueError("Cumulative frequency table must not be None")
        cum_freq = np.array(cum_freq)
        if cum_freq.ndim != 1:
            raise ValueError("Cumulative frequency table must be one-dimensional")
        freqs = np.empty_like(cum_freq)
        freqs[0] = cum_freq[0]
        freqs[1:] = cum_freq[1:] - cum_freq[:-1]
        total = cum_freq[-1]
        if total == 0:
            raise ValueError("Total frequency cannot be zero")
        probabilities = freqs / total
        return probabilities

    def pack_bits_to_bytes(self, bits: List[int]) -> bytes:
        """
        Pack a list of bits (0s and 1s) into a byte stream.
        
        Args:
            bits (List[int]): List of bits.
        
        Returns:
            bytes: Packed bytes.
        """
        if bits is None:
            raise ValueError("Bits cannot be None")
        byte_array = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = byte << 1
                if i + j < len(bits):
                    byte |= bits[i + j]
            byte_array.append(byte)
        return bytes(byte_array)

    def unpack_bytes_to_bits(self, data: bytes) -> List[int]:
        """
        Unpack a byte stream into a list of bits (MSB first).
        
        Args:
            data (bytes): The byte stream.
        
        Returns:
            List[int]: List of bits.
        """
        if data is None:
            raise ValueError("Data cannot be None")
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    def integer_arithmetic_encode_symbols(
        self,
        symbols: List[Symbol],
        prediction_model: BasePredictionModel,
        dictionary: Dictionary,
        state_bits: int = 32,
        train_callback: Optional[Callable[[List[Symbol], Symbol], None]] = None
    ) -> bytes:
        """
        Integer-based arithmetic encoding of symbols.
        
        Args:
            symbols (List[Symbol]): The symbols to encode.
            prediction_model (BasePredictionModel): The prediction model for probability estimation.
            dictionary (Dictionary): The symbol dictionary.
            state_bits (int): Number of bits for state representation.
            train_callback (Optional[Callable[[List[Symbol], Symbol], None]]): Callback for training.
        
        Returns:
            bytes: The encoded byte stream.
        """
        low = 0
        full_range = 1 << state_bits  
        half_range = full_range >> 1  
        quarter_range = half_range >> 1  
        high = full_range - 1

        out_buffer = BytesIO()
        bit_out = BitOutputStream(out_buffer)
        
        underflow = 0
        context: List[Symbol] = []

        sym_num = 0 # for logging
        for symbol in symbols:
            sym_num += 1 # for logging
            bits_before = out_buffer.tell() * 8 + bit_out.num_bits_filled  # for logging
            
            probs = prediction_model.predict(context)
            cum_freq = self.probabilities_to_code(probs)
            total = cum_freq[-1]

            symbol_index = dictionary.get_index(symbol)
            sym_low = cum_freq[symbol_index - 1] if symbol_index > 0 else 0
            sym_high = cum_freq[symbol_index]
            current_range = high - low + 1
            new_low = low + (sym_low * current_range) // total
            new_high = low + (sym_high * current_range) // total - 1
            low, high = new_low, new_high
            
            if self.logger is not None:
                self.logger.log(EncodedSymbolProbability(symbol, probs[symbol_index].frequency.item()))

            while True:
                if high < half_range:
                    bit_out.write(0)
                    for _ in range(underflow):
                        bit_out.write(1)
                    underflow = 0
                elif low >= half_range:
                    bit_out.write(1)
                    for _ in range(underflow):
                        bit_out.write(0)
                    underflow = 0
                    low -= half_range
                    high -= half_range
                elif low >= quarter_range and high < 3 * quarter_range:
                    underflow += 1
                    low -= quarter_range
                    high -= quarter_range
                else:
                    break
                low = (low << 1) & (full_range - 1)
                high = ((high << 1) & (full_range - 1)) | 1

            if train_callback is not None:
                Logger.log(PredictionModelTrainingProgressStep(len(context), len(symbols)))
                train_callback(context, symbol)
            context.append(symbol)
            
            bits_after = out_buffer.tell() * 8 + bit_out.num_bits_filled  # for logging
            encoded_bits = bits_after - bits_before  # for logging
            symbol_size = len(symbol.data) * 8  # for logging
            if self.logger is not None:
                self.logger.log(CodingLog(symbol_size, encoded_bits))
                self.logger.log(CodingProgressStep(sym_num, len(symbols)))

        underflow += 1
        if low < quarter_range:
            bit_out.write(0)
            for _ in range(underflow):
                bit_out.write(1)
        else:
            bit_out.write(1)
            for _ in range(underflow):
                bit_out.write(0)
        
        bit_out.finish()
        return out_buffer.getvalue()

    def integer_arithmetic_decode(
        self,
        bit_in: BitInputStream,
        num_symbols: int,
        prediction_model: BasePredictionModel,
        dictionary: Dictionary,
        state_bits: int = 32,
        train_callback: Optional[Callable[[List[Symbol], Symbol], None]] = None
    ) -> List[Symbol]:
        """
        Integer-based arithmetic decoding of symbols.
        
        Args:
            bit_in (BitInputStream): The input bit stream.
            num_symbols (int): Number of symbols to decode.
            prediction_model (BasePredictionModel): The prediction model for probability estimation.
            dictionary (Dictionary): The symbol dictionary.
            state_bits (int): Number of bits for state representation.
            train_callback (Optional[Callable[[List[Symbol], Symbol], None]]): Callback for training.
        
        Returns:
            List[Symbol]: The decoded symbols.
        """
        full_range = 1 << state_bits
        half_range = full_range >> 1
        quarter_range = half_range >> 1
        low = 0
        high = full_range - 1

        code = 0
        for _ in range(state_bits):
            next_bit = bit_in.read()
            code = (code << 1) | (next_bit if next_bit != -1 else 0)

        decoded_symbols: List[Symbol] = []
        context: List[Symbol] = []
        for _ in range(num_symbols):
            probs = prediction_model.predict(context)
            cum_freq = self.probabilities_to_code(probs)
            total = cum_freq[-1]
            current_range = high - low + 1
            scaled_value = ((code - low + 1) * total - 1) // current_range

            symbol_index: Optional[int] = None
            for i, freq in enumerate(cum_freq):
                if scaled_value < freq:
                    symbol_index = i
                    break
            if symbol_index is None:
                raise ValueError("Failed to decode symbol: no matching frequency range found.")
            
            decoded_symbol = dictionary.get_symbol_by_index(symbol_index)
            
            if train_callback is not None:
                train_callback(context, decoded_symbol)
            decoded_symbols.append(decoded_symbol)
            context.append(decoded_symbol)

            sym_low = cum_freq[symbol_index - 1] if symbol_index > 0 else 0
            sym_high = cum_freq[symbol_index]
            new_low = low + (sym_low * current_range) // total
            new_high = low + (sym_high * current_range) // total - 1
            low, high = new_low, new_high

            while True:
                if high < half_range:
                    pass
                elif low >= half_range:
                    low -= half_range
                    high -= half_range
                    code -= half_range
                elif low >= quarter_range and high < 3 * quarter_range:
                    low -= quarter_range
                    high -= quarter_range
                    code -= quarter_range
                else:
                    break
                low = (low << 1) & (full_range - 1)
                high = ((high << 1) & (full_range - 1)) | 1
                next_bit = bit_in.read()
                code = ((code << 1) & (full_range - 1)) | (next_bit if next_bit != -1 else 0)

            if self.logger is not None:
                self.logger.log(EncodedSymbolProbability(decoded_symbol, probs[symbol_index].frequency.item()))
                self.logger.log(CodingProgressStep(bit_in.inp.tell() * 8, num_symbols))

        return decoded_symbols


class ArithmeticCoderOnline(CoderBase):
    """
    Online arithmetic coder that updates the prediction model during encoding/decoding.
    """

    def __init__(self, arithmetic_coder_settings: ArithmeticCoderSettings, logger: Optional[Logger] = None) -> None:
        validate_type(arithmetic_coder_settings, "arithmetic_coder_settings", ArithmeticCoderSettings)
        self.state_bits: int = 32
        self.codec: ArithmeticCodec = ArithmeticCodec(arithmetic_coder_settings, logger)
        self.coder_code: int = 1
        self.logger: Optional[Logger] = logger

    def encode(self, symbols: List[Symbol], prediction_model: BasePredictionModel) -> bytes:
        """
        Encode symbols using the online arithmetic coder.
        
        Args:
            symbols (List[Symbol]): The symbols to encode.
            prediction_model (BasePredictionModel): The prediction model.
        
        Returns:
            bytes: The encoded data.
        """
        if symbols is None or not isinstance(symbols, list) or len(symbols) == 0:
            raise ValueError("Symbols must be a non-empty list of Symbol objects")
        num_symbols: int = len(symbols)
        header: bytes = num_symbols.to_bytes(4, byteorder="big")
        bitstream_bytes: bytes = self.codec.integer_arithmetic_encode_symbols(
            symbols,
            prediction_model,
            prediction_model.dictionary,
            self.state_bits,
            train_callback=lambda ctx, sym: prediction_model.train(ctx, sym)
        )
        return header + bitstream_bytes

    def decode(self, data: bytes, dictionary: Dictionary, prediction_model: BasePredictionModel) -> List[Symbol]:
        """
        Decode data using the online arithmetic coder.
        
        Args:
            data (bytes): The encoded data.
            dictionary (Dictionary): The symbol dictionary.
            prediction_model (BasePredictionModel): The prediction model.
        
        Returns:
            List[Symbol]: The decoded symbols.
        """
        validate_type(data, "data", bytes)
        if len(data) < 4:
            raise ValueError("Data must be a byte string with at least 4 bytes for the header")
        if dictionary is None:
            raise ValueError("Dictionary cannot be None")
        header: bytes = data[:4]
        num_symbols: int = int.from_bytes(header, byteorder="big")
        bitstream_data: bytes = data[4:]
        in_buffer: BytesIO = BytesIO(bitstream_data)
        bitstream: BitInputStream = BitInputStream(in_buffer)
        decoded_symbols: List[Symbol] = self.codec.integer_arithmetic_decode(
            bitstream,
            num_symbols,
            prediction_model,
            dictionary,
            self.state_bits,
            train_callback=lambda ctx, sym: prediction_model.train(ctx, sym)
        )
        return decoded_symbols

    def get_coder_code(self) -> int:
        """
        Get the coder code.
        
        Returns:
            int: The code (1 for online coder).
        """
        return self.coder_code


class ArithmeticCoderOffline(CoderBase):
    """
    Offline arithmetic coder that trains the prediction model prior to encoding.
    """

    def __init__(self, coder_settings: ArithmeticCoderSettings, logger: Optional[Logger] = None) -> None:
        validate_type(coder_settings, "coder_settings", ArithmeticCoderSettings)
        self.state_bits: int = 32
        self.codec: ArithmeticCodec = ArithmeticCodec(coder_settings, logger)
        self.logger: Optional[Logger] = logger
        self.coder_code: int = 2

    def encode(self, input_symbols: List[Symbol], prediction_model: BasePredictionModel) -> bytes:
        """
        Encode symbols using the offline arithmetic coder.
        
        Args:
            input_symbols (List[Symbol]): The symbols to encode.
            prediction_model (BasePredictionModel): The prediction model (must have train and save_model methods and a dictionary attribute).
        
        Returns:
            bytes: The encoded data, including model weights and the arithmetic bitstream.
        """
        if not isinstance(input_symbols, list):
            raise ValueError("Input symbols must be a list.")
        if prediction_model is None:
            raise ValueError("Prediction model cannot be None.")
        if not hasattr(prediction_model, "train") or not hasattr(prediction_model, "save_model"):
            raise ValueError("Prediction model must have train and save_model methods.")
        if not hasattr(prediction_model, "dictionary"):
            raise ValueError("Prediction model must have a dictionary attribute.")

        for i, symbol in enumerate(input_symbols):
            context: List[Symbol] = input_symbols[:i]
            prediction_model.train(context, symbol)
            if self.logger is not None:
                self.logger.log(PredictionModelTrainingProgressStep(i, len(input_symbols)))
        
        temp_dir: str = tempfile.gettempdir()
        temp_filename: str = next(tempfile._get_candidate_names()) + ".weights.h5"
        temp_filepath: str = os.path.join(temp_dir, temp_filename)
        prediction_model.save_model(temp_filepath)
        if not os.path.exists(temp_filepath):
            raise IOError("Model weights file was not created.")
        with open(temp_filepath, "rb") as f:
            weights_data: bytes = f.read()
        os.remove(temp_filepath)
        weights_header: bytes = struct.pack(">Q", len(weights_data))
        
        msg_length_header: bytes = struct.pack(">I", len(input_symbols))
        arithmetic_bitstream: bytes = self.codec.integer_arithmetic_encode_symbols(
            input_symbols,
            prediction_model,
            prediction_model.dictionary,
            self.state_bits,
            train_callback=None
        )
        arithmetic_bitstream = msg_length_header + arithmetic_bitstream
        
        return weights_header + weights_data + arithmetic_bitstream

    def decode(self, encoded_data: bytes, dictionary: Dictionary, prediction_model: BasePredictionModel) -> List[Symbol]:
        """
        Decode data using the offline arithmetic coder.
        
        Args:
            encoded_data (bytes): The encoded data.
            dictionary (Dictionary): The symbol dictionary.
            prediction_model (BasePredictionModel): The prediction model.
        
        Returns:
            List[Symbol]: The decoded symbols.
        """
        validate_type(encoded_data, "encoded_data", bytes)
        if len(encoded_data) < 8:
            raise ValueError("Encoded data is too short.")
        if dictionary is None or prediction_model is None:
            raise ValueError("Dictionary and prediction model cannot be None.")
        
        weights_header: bytes = encoded_data[:8]
        weights_length: int = struct.unpack(">Q", weights_header)[0]
        if len(encoded_data) < 8 + weights_length:
            raise ValueError("Encoded data is missing weights.")
        weights_data: bytes = encoded_data[8:8+weights_length]
        arithmetic_data: bytes = encoded_data[8+weights_length:]
        
        temp_dir: str = tempfile.gettempdir()
        temp_filename: str = next(tempfile._get_candidate_names()) + ".weights.h5"
        temp_filepath: str = os.path.join(temp_dir, temp_filename)
        with open(temp_filepath, "wb") as f:
            f.write(weights_data)
        prediction_model.load_model(temp_filepath)
        os.remove(temp_filepath)
        
        if len(arithmetic_data) < 4:
            raise ValueError("Arithmetic data missing message length header.")
        msg_length_header: bytes = arithmetic_data[:4]
        message_length: int = struct.unpack(">I", msg_length_header)[0]
        bitstream_bytes: bytes = arithmetic_data[4:]
        in_buffer: BytesIO = BytesIO(bitstream_bytes)
        bitstream: BitInputStream = BitInputStream(in_buffer)
        decoded_symbols: List[Symbol] = self.codec.integer_arithmetic_decode(
            bitstream,
            message_length,
            prediction_model,
            dictionary,
            self.state_bits,
            train_callback=None
        )
        return decoded_symbols

    def get_coder_code(self) -> int:
        """
        Get the coder code.
        
        Returns:
            int: The code (2 for offline coder).
        """
        return self.coder_code


def get_coder(code: int, logger: Optional[Logger] = None) -> CoderBase:
    """
    Retrieve a coder instance based on the given code.
    
    Args:
        code (int): The coder code (1 for online, 2 for offline).
        logger (Optional[Logger]): Logger instance for logging.
    
    Returns:
        CoderBase: An instance of a coder.
    
    Raises:
        ValueError: If the coder code is unknown.
    """
    if code == 1:
        return ArithmeticCoderOnline(ArithmeticCoderSettings(), logger)
    elif code == 2:
        return ArithmeticCoderOffline(ArithmeticCoderSettings(), logger=logger)
    else:
        raise ValueError("Unknown coder code: " + str(code))
