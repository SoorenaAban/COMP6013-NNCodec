#loger.py

from datetime import datetime

from .models import *

#log level enum
class LogLevel:
    INFO = 0
    WARNING = 1
    ERROR = 2

class Log:
    def __init__(self, type_name, level, message):
        self.level = level
        self.type_name = type_name
        self.message = message
        self.date = datetime.now()
        
        
    def __str__(self):
        return f"{self.date} - {self.type_name} - {self.level} - {self.message}"
    
    def __repr__(self):
        return self.__str__()
    
class CodingLog(Log):
    def __init__(self, symbol_size, encoded_size):
        
        self.symbol_size = symbol_size
        self.encoded_size = encoded_size
        
        super().__init__("Coding_log", LogLevel.INFO, f"Symbol size: {symbol_size}, Encoded size: {encoded_size}")
        
class PredictionModelProabilityLog(Log):
    def __init__(self, symbol, prob):
        self.symbol = symbol
        self.prob = prob
        super().__init__("Prediction_model_log", LogLevel.INFO, f"Symbol: {symbol}, Probability: {prob}")
        
class PredictionModelTrainingLog(Log):
    def __init__(self, loss):
        self.loss = loss
        super().__init__("Prediction_model_training_log", LogLevel.INFO, f"Loss: {loss}")

class Logger:
    def __init__(self):
        self.logs = []
        
        self.recod_info = True
        self.record_warning = True
        self.record_error = True
        
        self.display_info = True
        self.display_warning = True
        self.display_error = True
        
        self.save_info = True
        self.save_warning = True
        self.save_error = True
        
        
        pass

    def log(self, log):
        if (not isinstance(log, Log) or not isinstance(log, str)):
            raise ValueError("Log must be an instance of Log class or a string")
        if isinstance(log, str):
            log = Log("General", LogLevel.INFO, log)
        
        if log.level == LogLevel.INFO:
            if self.recod_info:
                self.logs.append(log)
            if self.display_info:
                print(log)
        elif log.level == LogLevel.WARNING:
            if self.record_warning:
                self.logs.append(log)
            if self.display_warning:
                print(log)
        elif log.level == LogLevel.ERROR:
            if self.record_error:
                self.logs.append(log)
            if self.display_error:
                print(log)
                
    def save(self, file_path):
        pass