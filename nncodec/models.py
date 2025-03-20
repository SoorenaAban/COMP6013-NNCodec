#models.py

import uuid
import time
from enum import Enum


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
    """Dictionary class represents the dictionary of symbols in the data"""
    def __init__(self):
        self.symbols = set()

    def add(self, symbol):
        if symbol in self.symbols:
            return True
        self.symbols.add(symbol)
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
            if symbol.code == code:
                return True
        return False
