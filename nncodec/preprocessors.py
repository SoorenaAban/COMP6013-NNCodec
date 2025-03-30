import abc
from typing import List, Tuple, Optional

from .models import Symbol, Dictionary
from .logger import Logger, PreprocessingProgressStep


class BasePreprocessor(abc.ABC):
    @property
    @abc.abstractmethod
    def code(self) -> int:
        """Return the unique identification code for the preprocessor."""
        pass

    @property
    @abc.abstractmethod
    def header_size(self) -> int:
        """Return the number of bytes required for the header."""
        pass

    @abc.abstractmethod
    def convert_to_symbols(self, data: bytes) -> Tuple[List[Symbol], Dictionary]:
        """
        Convert raw data (bytes) to a list of symbols and construct a dictionary.
        
        Args:
            data (bytes): The input data as bytes.
        
        Returns:
            Tuple[List[Symbol], Dictionary]: A tuple containing the list of symbols and the constructed dictionary.
        """
        pass

    @abc.abstractmethod
    def convert_from_symbols(self, symbols: List[Symbol]) -> bytes:
        """
        Convert a list of symbols back to data in bytes.
        
        Args:
            symbols (List[Symbol]): The list of symbols.
        
        Returns:
            bytes: The reconstructed data.
        """
        pass

    def construct_dictionary_from_symbols(self, symbols: List[Symbol]) -> Dictionary:
        """
        Construct a dictionary from a list of symbols.
        
        Args:
            symbols (List[Symbol]): A collection of symbols.
        
        Returns:
            Dictionary: A dictionary built from the provided symbols.
        """
        dictionary = Dictionary()
        dictionary.add_multiple(symbols)
        return dictionary

    @abc.abstractmethod
    def encode_dictionary_for_header(self, dictionary: Dictionary) -> bytes:
        """
        Convert a dictionary into its binary representation to be stored in a header.
        
        Args:
            dictionary (Dictionary): The dictionary to encode.
        
        Returns:
            bytes: The binary representation of the dictionary.
        """
        pass

    @abc.abstractmethod
    def construct_dictionary_from_header(self, data: bytes) -> Dictionary:
        """
        Construct a dictionary from header data.
        
        Args:
            data (bytes): The header data.
        
        Returns:
            Dictionary: The reconstructed dictionary.
        """
        pass


class AsciiCharPreprocessor(BasePreprocessor):
    """
    ASCII Character Preprocessor: Each ASCII character is assigned to a symbol.
    """
    def __init__(self, logger: Optional[Logger] = None) -> None:
        self.logger: Optional[Logger] = logger

    @property
    def header_size(self) -> int:
        return 16

    @property
    def code(self) -> int:
        return 4

    def convert_to_symbols(self, data: bytes) -> Tuple[List[Symbol], Dictionary]:
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        
        try:
            data.decode('ascii')
        except UnicodeDecodeError:
            raise ValueError("Data should contain only ASCII characters")
        
        uppercase_flag_symbol = Symbol(b'\x82')
        dictionary = Dictionary()
        symbols: List[Symbol] = []
        for byte in data:
            if 65 <= byte <= 90:
                symbols.append(Symbol(b'\x82'))
                lower_byte = byte + 32
                symbols.append(Symbol(bytes([lower_byte])))
                if not dictionary.contains_data(bytes([lower_byte])):
                    dictionary.add(Symbol(bytes([lower_byte])))
            else:
                symbols.append(Symbol(bytes([byte])))
                if not dictionary.contains_data(bytes([byte])):
                    dictionary.add(Symbol(bytes([byte])))
        
        if not dictionary.contains_data(uppercase_flag_symbol.data):
            dictionary.add(uppercase_flag_symbol)
        
        return symbols, dictionary

    def convert_from_symbols(self, symbols: List[Symbol]) -> bytes:
        data = b''
        uppercase_flag = False
        for symbol in symbols:
            if symbol.data == b'\x82':
                uppercase_flag = True
            else:
                if uppercase_flag:
                    char = symbol.data.decode('ascii').upper()
                    data += char.encode('ascii')
                    uppercase_flag = False
                else:
                    data += symbol.data
        return data

    def encode_dictionary_for_header(self, dictionary: Dictionary) -> bytes:
        header_int = 0
        # Exclude the uppercase flag symbol from the header encoding.
        symbols_to_encode = [s for s in dictionary.symbols if s.data != b'\x82']
        for symbol in symbols_to_encode:
            char_val = symbol.data[0]
            header_int |= (1 << char_val)
        return header_int.to_bytes(16, 'little')

    def construct_dictionary_from_header(self, data: bytes) -> Dictionary:
        if len(data) != 16:
            raise ValueError("Header data must be exactly 16 bytes")
        header_int = int.from_bytes(data, 'little')
        dictionary = Dictionary()
        for i in range(128):
            if header_int & (1 << i):
                dictionary.add(Symbol(chr(i).encode('ascii')))
        uppercase_flag_symbol = Symbol(b'\x82')
        if not dictionary.contains_data(uppercase_flag_symbol.data):
            dictionary.add(uppercase_flag_symbol)
        return dictionary
        

class BytePreprocessor(BasePreprocessor):
    """
    Byte Preprocessor: Each byte of data is assigned to a symbol.
    """
    def __init__(self, logger: Optional[Logger] = None) -> None:
        self.logger: Optional[Logger] = logger

    @property
    def header_size(self) -> int:
        return 32

    @property
    def code(self) -> int:
        return 3

    def convert_to_symbols(self, data: bytes) -> Tuple[List[Symbol], Dictionary]:
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        
        dictionary = Dictionary()
        symbols: List[Symbol] = []
        cache = {}
        for b in data:
            if b not in cache:
                byte_data = bytes([b])
                symbol = Symbol(byte_data)
                cache[b] = symbol
                dictionary.add(symbol)
            symbols.append(cache[b])
            if self.logger is not None:
                self.logger.log(PreprocessingProgressStep("Converting data to symbols", len(data)))
    
        return symbols, dictionary

    def convert_from_symbols(self, symbols: List[Symbol]) -> bytes:
        data = b''
        for symbol in symbols:
            data += symbol.data
        return data

    def encode_dictionary_for_header(self, dictionary: Dictionary) -> bytes:
        header_int = 0
        for symbol in dictionary.symbols:
            byte_val = symbol.data[0]
            header_int |= (1 << byte_val)
        return header_int.to_bytes(32, 'little')

    def construct_dictionary_from_header(self, data: bytes) -> Dictionary:
        if len(data) != 32:
            raise ValueError("Header data must be exactly 32 bytes")
        header_int = int.from_bytes(data, 'little')
        dictionary = Dictionary()
        for i in range(256):
            if header_int & (1 << i):
                dictionary.add(Symbol(i.to_bytes(1, 'big')))
        return dictionary


def get_preprocessor(code: int, logger: Optional[Logger] = None) -> BasePreprocessor:
    """
    Retrieve a preprocessor instance based on the given code.
    
    Args:
        code (int): The preprocessor code.
        logger (Optional[Logger]): Logger instance for logging.
    
    Returns:
        BasePreprocessor: An instance of a preprocessor.
    
    Raises:
        ValueError: If the preprocessor code is not supported.
    """
    if code == 3:
        return BytePreprocessor(logger)
    elif code == 4:
        return AsciiCharPreprocessor(logger)
    else:
        raise ValueError("Preprocessor code not supported")
