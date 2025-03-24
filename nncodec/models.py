#models.py

import uuid
import time
from enum import Enum

class CompressionResult:
    def __init__(self, success, compressed_data_size, compression_ratio, compression_time, decompression_time):
        self.compression_decompression_success = False

class Symbol:
    """ Symbol class represents a single symbol in the data"""
    def __init__(self, data):
        if not isinstance(data, bytes):
            raise ValueError("Data should be in form of bytes")
        self.data = data

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.data == other.data
        return False
    
    def __str__(self):
        try :
            return str(self.data)
        except:
            return f"[Symbol:{self.__hash__()}]"
    
    def __repr__(self):
        return str(self.data)

    def __hash__(self):
        return hash(self.data)

class SymbolFrequency:
    """ SymbolFrequency class represents a single symbol in the data with its frequency"""
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        
    def __str__(self):
        return f"[{self.symbol},{self.frequency}]"

    def __repr__(self):
        return f"[{self.symbol},{self.frequency}]"

class Dictionary:
    """Dictionary class represents the dictionary of symbols in the data."""
    def __init__(self):
        self.symbols = set()
        self._sorted_symbols = None
        self._symbol_to_index = None

    def add(self, symbol):
        if symbol in self.symbols:
            return True
        self.symbols.add(symbol)
        # Invalidate the cached sorted list and mapping
        self._sorted_symbols = None
        self._symbol_to_index = None
        return False

    def add_multiple(self, symbols):
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
            if hasattr(symbol, 'code') and symbol.code == code:
                return True
        return False

    def _build_index(self):
        """Sort symbols and build a mapping from symbol to its index."""
        self._sorted_symbols = sorted(self.symbols, key=lambda s: s.data)
        self._symbol_to_index = {s: i for i, s in enumerate(self._sorted_symbols)}

    def get_index(self, symbol):
        """Return the index of the given symbol from the sorted list."""
        if self._symbol_to_index is None:
            self._build_index()
        return self._symbol_to_index[symbol]

    def get_sorted_symbols(self):
        """Return the sorted list of symbols."""
        if self._sorted_symbols is None:
            self._build_index()
        return self._sorted_symbols
    
    def get_symbol_by_index(self, index):
        """Return the symbol at the given index from the sorted symbols list."""
        return self.get_sorted_symbols()[index]
    
    def __eq__(self, value):
        if not isinstance(value, Dictionary):
            return False
        if len(self.symbols) != len(value.symbols):
            return False
        for symbol in self.symbols:
            if not value.contains(symbol):
                return False
