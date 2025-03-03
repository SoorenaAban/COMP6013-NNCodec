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

class CompressedModel:
    """To be used to represent the compressed file"""
    def __init__(self):
        self.vocab_code = None # 2 bytes
        self.version = None # 2 bytes
        self.preprocessor_header = None
        self.data = None
        pass

class LogType(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3
    
class ModuleStage(Enum):
    COMPRESSION = 1
    DECOMPRESSION = 2
    
class LogProgress:
    """"""
    def __init__(self, step, stage, total_steps = None):
        if(not isinstance(step, int)):
            raise ValueError("Step should be of type int")
        if(not isinstance(stage, ModuleStage)):
            raise ValueError("Stage should be of type ModuleStage")
        if(total_steps is not None and not isinstance(total_steps, int)):
            raise ValueError("Total steps should be of type int")
        
        self.step = step
        self.stage = stage
        self.total_steps = total_steps
        self.timestamp = time.time()
        pass

class LogMessage:
    """To be used to log the compression and decompression process"""
    def __init__(self, log_type, message, progress, tags=[]):
        if not isinstance(log_type, LogType):
            raise ValueError("Log type should be of type LogType")
        if not isinstance(message, str):
            raise ValueError("Message should be of type string")
        if not isinstance(progress, LogProgress):
            raise ValueError("Progress should be of type LogProgress")
        if (not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags)):
            if isinstance(tags, str):
                self.tags = []
                for tag in tags.split(","):
                    self.tags.append(tag.strip())
            else:
                raise ValueError("Tags should be of type list of strings")
            raise ValueError("Tags should be of type list of strings")
        else:
            self.tags = tags
        self.timestamp = time.time()
        self.log_type = log_type
        self.messasge = message
        self.progress = progress

    def __str__(self):
        return f"[{self.log_type}]({self.progress}):{self.messasge} | {self.tags}"

    def __repr__(self):
        return f"[{self.log_type}]({self.progress}):{self.messasge} | {self.tags}"