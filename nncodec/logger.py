#loger.py

from datetime import datetime
import os

from .models import *

#log level enum
class LogLevel:
    INFO = 0
    WARNING = 1
    ERROR = 2
    PROGRESS = 3 #probaly shouldn't have it in logging...

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

class PreprocessingSymbolCreationLog(Log):
    def __init__(self, symbol):
        self.symbol = symbol
        super().__init__("Preprocessing_symbol_creation_log", LogLevel.INFO, f"Symbol: {symbol}")
    
class CodingLog(Log):
    def __init__(self, symbol_size, encoded_size):
        
        self.symbol_size = symbol_size
        self.encoded_size = encoded_size
        
        super().__init__("Coding_log", LogLevel.INFO, f"Symbol size: {symbol_size}, Encoded size: {encoded_size}")
        
class EncodedSymbolProbability(Log):
    def __init__(self, symbol, prob):
        self.symbol = symbol
        self.prob = prob
        super().__init__("EncodedSymbolProbability", LogLevel.INFO, f"Symbol: {symbol}, Probability: {prob}")
        
class PredictionModelTrainingLog(Log):
    def __init__(self, loss):
        self.loss = loss
        super().__init__("Prediction_model_training_log", LogLevel.INFO, f"Loss: {loss}")
        

        
        
class PreprocessingProgressStep(Log):
    
    TotalSteps = None
    CountedSteps = None
    
    def __init__(self, message, total_steps = None):
        if total_steps is not None:
            PreprocessingProgressStep.TotalSteps = total_steps
            
        if PreprocessingProgressStep.CountedSteps is None:
            PreprocessingProgressStep.CountedSteps = 0
            
        PreprocessingProgressStep.CountedSteps += 1
        
        if PreprocessingProgressStep.TotalSteps is not None:
            self.message = f"{message} ({PreprocessingProgressStep.CountedSteps}/{PreprocessingProgressStep.TotalSteps})"
        else:
            self.message = f"{message} ({PreprocessingProgressStep.CountedSteps})"
        
        super().__init__("Preprocessing_progress_step", LogLevel.PROGRESS, self.message)
        
        
class CodingProgressStep(Log):
    
    TotalSteps = None
    CountedSteps = None
    
    def __init__(self, message, total_steps = None):
        if total_steps is not None:
            CodingProgressStep.TotalSteps = total_steps
            
        if CodingProgressStep.CountedSteps is None:
            CodingProgressStep.CountedSteps = 0
            
        CodingProgressStep.CountedSteps += 1
        
        if CodingProgressStep.TotalSteps is not None:
            self.message = f"{message} ({CodingProgressStep.CountedSteps}/{CodingProgressStep.TotalSteps})"
        else:
            self.message = f"{message} ({CodingProgressStep.CountedSteps})"
        
        super().__init__("Coding_progress_step", LogLevel.PROGRESS, self.message)
        
class PredictionModelTrainingProgressStep(Log):
    TotalSteps = None
    CountedSteps = None
    
    def __init__(self, message, total_steps = None):
        if total_steps is not None:
            PredictionModelTrainingProgressStep.TotalSteps = total_steps
            
        if PredictionModelTrainingProgressStep.CountedSteps is None:
            PredictionModelTrainingProgressStep.CountedSteps = 0
            
        PredictionModelTrainingProgressStep.CountedSteps += 1
        
        if PredictionModelTrainingProgressStep.TotalSteps is not None:
            self.message = f"{message} ({PredictionModelTrainingProgressStep.CountedSteps}/{PredictionModelTrainingProgressStep.TotalSteps})"
        else:
            self.message = f"{message} ({PredictionModelTrainingProgressStep.CountedSteps})"
        
        super().__init__("Prediction_model_training_progress_step", LogLevel.PROGRESS, self.message)

class Logger:
    def __init__(self):
        self.logs = []
        
        self.record_info = True
        self.record_warning = True
        self.record_error = True
        self.record_progress = False
        
        self.display_info = False
        self.display_warning = True
        self.display_error = True
        self.display_progress = True
        
        self.save_info = True
        self.save_warning = True
        self.save_error = True
        self.save_progress = False
        
        self.preprocessor_step_interval_count = 10000
        self.coding_step_interval_count = 1000
        self.training_step_interval_count = 1000

    def log(self, log):
        if not (isinstance(log, Log) or isinstance(log, str)):
            raise ValueError("Log must be an instance of Log class or a string")
        if isinstance(log, str):
            log = Log("General", LogLevel.INFO, log)
        
        if log.level == LogLevel.INFO:
            if self.record_info:
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
                
        elif log.level == LogLevel.PROGRESS:
            if self.record_progress:
                self.logs.append(log)
            if self.display_progress:
                if isinstance(log, PreprocessingProgressStep):   
                    if (PreprocessingProgressStep.CountedSteps % self.preprocessor_step_interval_count) == 0:
                        print(log)
                        
                elif isinstance(log, CodingProgressStep):
                    if (CodingProgressStep.CountedSteps % self.coding_step_interval_count) == 0:
                        print(log)
                        
                elif isinstance(log, PredictionModelTrainingProgressStep):
                    if (PredictionModelTrainingProgressStep.CountedSteps % self.training_step_interval_count) == 0:
                        print(log)
            
    def save(self, file_path):
        with open(file_path, 'w') as file:
            for log in self.logs:
                file.write(str(log) + "\n")