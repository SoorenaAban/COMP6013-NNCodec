#models.py

import uuid

class Symbol:
    """ Symbol class represents a single symbol in the data"""
    def __init__(self, data):
        """data: the data of the symbol. It should be in the form of bytes"""
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        self.data = data
        self.id = uuid.uuid4()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.id)

class SymbolFrequency:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency

class SymbolsFrequencies:
    def __init__(self):
        self.symbol_frequencies = []

class Dictionary:
    def __init__(self):
        self.symbols = set()

    def add(self, symbol):
        """adds a symbol to the dictionary
        symbol: the symbol to be added
        returns True if the symbol is already in the dictionary, False otherwise"""
        if symbol in self.symbols:
            return True
        self.symbols.add(symbol)
        return False

    def add_multiple(self, symbols):
        """adds multiple symbols to the dictionary
        symbols: the symbols to be added
        returns the number of symbols added"""
        count = 0
        for symbol in symbols:
            if self.add(symbol):
                count += 1
        return count

    def get_size(self):
        return len(self.symbols)

    def contains(self, symbol):
        return symbol in self.symbols
    def contains_data(self, data):
        for symbol in self.symbols:
            if symbol.data == data:
                return True
        return False
    def contains_code(self, code):
        for symbol in self.symbols:
            if symbol.code == code:
                return True
        return False

class CompressedModel:
    def __init__(self):
        self.length = 0 # 6 byes
        self.vocab_code = None # 2 bytes
        self.version = None # 2 bytes
        self.data = None
        pass