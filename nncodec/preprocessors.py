#preprocessors.py

import abc

from .models import Dictionary, Symbol

class base_preprocessor(abc.ABC):
    @property
    @abc.abstractmethod
    def code(self):
        """The code of the preprocessor for identification"""
        pass

    @property
    @abc.abstractmethod
    def header_size(self):
        """the amount of bytes required for the header"""
        pass

    # @abc.abstractmethod
    # def get_terminals(self):
    #     """returns the header terminator in form of bytes, in case no set size for the header is present"""
    #     pass

    @abc.abstractmethod
    def convert_to_symbols(self, data):
        """ 
        Convert data to symbols and constructs dictionary

        Args:
            data: a data object, should be in form of bytes.

        Returns:
            list[symbols]: a colection of symbols representing the data
            Dictionary: a dictionary of symbols

        """
        pass

    @abc.abstractmethod
    def convert_from_symbols(self, data):
        """ 
        Convert symbols back to data

        Args:
            data(list[Symbol]): a collection of symbols

        Returns:
            list[byte]: decoded data in form of bytes
        """
        pass

    def construct_dictionary_from_symbols(self, symbols):
        """ 
        Construct a dictionary of symbols and their corresponding probabilities from a set of symbols
        maybe 

        Args:
            symbols: a collection of symbols

        Returns:
            _type_: a dictionary of symbols and their corresponding probabilities
        """
        dictionary = Dictionary()
        dictionary.add_multiple(symbols)
        return dictionary

    @abc.abstractmethod
    def encode_dictionary_for_header(self, dictionary):
        """
        turn dictionary into a binary representation to be written to the header in form of bytes
        
        Args:
            dictionary(Dictionary): the dictionary to be encoded
        
        Returns:
            list[byte]: the binary representation of the dictionary
        """
        pass

    @abc.abstractmethod
    def construct_dictionary_from_header(self, data):
        """construct a dictionary of symbols from the header data and based on the preprocessor type"""
        pass

# class charPreprocessor(base_preprocessor):
#     def convert_to_symbols(self, data):
#         """ Convert text to binary representation

#         Args:
#             data (string): a data object, in form of binary.

#         Returns:
#             _type_: a colelction of symbols representing the data
#         """

#         symbols = []
#         for char in data:
#             symbols.append(models.Symbol(char))

#     def convert_from_symbols(self, symbols):
#         """ Convert symbols back to text

#         Args:
#             symbols: a collection of symbols

#         Returns:
#             _type_: the data object in bytes
#         """
#         data = b''
#         for symbol in symbols:
#             data += symbol.data
#         return data

class byte_preprocessor(base_preprocessor):
    ''' Byte Preprocessor: each byte of data is assigned to a symbol. '''
    def __init__(self):
        pass

    @property
    def header_size(self):
        return 32

    @property
    def code(self):
        return 3

    def convert_to_symbols(self, data):
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        
        dictionary = Dictionary()
        symbols = []
        for byte in data:
            byte_data = bytes([byte])
            symbol = Symbol(byte_data)
            symbols.append(symbol)
            if not dictionary.contains_data(byte_data):
                dictionary.add(symbol)
        return symbols, dictionary

    def convert_from_symbols(self, symbols):
        data = b''
        for symbol in symbols:
            data += symbol.data
        return data

    def encode_dictionary_for_header(self, dictionary):
        header_int = 0
        for symbol in dictionary.symbols:
            byte_val = symbol.data[0]
            header_int |= (1 << byte_val)
        return header_int.to_bytes(32, 'little')

    def construct_dictionary_from_header(self, data):
        if len(data) != 32:
            raise ValueError("Header data must be exactly 32 bytes")
        header_int = int.from_bytes(data, 'little')
        dictionary = Dictionary()
        for i in range(256):
            if header_int & (1 << i):
                dictionary.add(Symbol(i.to_bytes(1, 'big')))
        return dictionary
