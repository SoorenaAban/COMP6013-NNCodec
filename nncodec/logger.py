"""
logger.py

Logging module for nncodec.


"""


from datetime import datetime
from typing import Union, Optional

class LogLevel:
    INFO = 0
    WARNING = 1
    ERROR = 2
    PROGRESS = 3 


class Log:
    def __init__(self, type_name: str, level: int, message: str) -> None:
        self.level = level
        self.type_name = type_name
        self.message = message
        self.date = datetime.now()
        
    def __str__(self) -> str:
        return f"{self.date} - {self.type_name} - {self.level} - {self.message}"
    
    def __repr__(self) -> str:
        return self.__str__()


class PreprocessingSymbolCreationLog(Log):
    def __init__(self, symbol: any) -> None:
        self.symbol = symbol
        super().__init__("Preprocessing_symbol_creation_log", LogLevel.INFO, f"Symbol: {symbol}")


class CodingLog(Log):
    def __init__(self, symbol_size: int, encoded_size: int) -> None:
        self.symbol_size = symbol_size
        self.encoded_size = encoded_size
        super().__init__("Coding_log", LogLevel.INFO, f"Symbol size: {symbol_size}, Encoded size: {encoded_size}")


class EncodedSymbolProbability(Log):
    def __init__(self, symbol: any, prob: float) -> None:
        self.symbol = symbol
        self.prob = prob
        super().__init__("EncodedSymbolProbability", LogLevel.INFO, f"Symbol: {symbol}, Probability: {prob}")


class PredictionModelTrainingLog(Log):
    def __init__(self, loss: float) -> None:
        self.loss = loss
        super().__init__("Prediction_model_training_log", LogLevel.INFO, f"Loss: {loss}")


class PreprocessingProgressStep(Log):
    def __init__(self, message: str, total_steps: Optional[int] = None) -> None:
        self.base_message = message
        self.total_steps = total_steps
        super().__init__("Preprocessing_progress_step", LogLevel.PROGRESS, message)


class CodingProgressStep(Log):
    def __init__(self, message: str, total_steps: Optional[int] = None) -> None:
        self.base_message = message
        self.total_steps = total_steps
        super().__init__("Coding_progress_step", LogLevel.PROGRESS, message)


class PredictionModelTrainingProgressStep(Log):
    def __init__(self, message: str, total_steps: Optional[int] = None) -> None:
        self.base_message = message
        self.total_steps = total_steps
        super().__init__("Prediction_model_training_progress_step", LogLevel.PROGRESS, message)


class Logger:
    def __init__(self) -> None:
        self.preproc_progress_count = 0
        self.coding_progress_count = 0
        self.training_progress_count = 0

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
        self.coding_step_interval_count = 100
        self.training_step_interval_count = 100

    def log(self, log: Union[Log, str]) -> None:
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
            if isinstance(log, PreprocessingProgressStep):
                self.preproc_progress_count += 1
                count = self.preproc_progress_count
                if log.total_steps is not None:
                    log.message = f"{log.base_message} ({count}/{log.total_steps})"
                else:
                    log.message = f"{log.base_message} ({count})"
                if self.record_progress:
                    self.logs.append(log)
                if self.display_progress and (count % self.preprocessor_step_interval_count == 0):
                    print(log)
            elif isinstance(log, CodingProgressStep):
                self.coding_progress_count += 1
                count = self.coding_progress_count
                if log.total_steps is not None:
                    log.message = f"{log.base_message} ({count}/{log.total_steps})"
                else:
                    log.message = f"{log.base_message} ({count})"
                if self.record_progress:
                    self.logs.append(log)
                if self.display_progress and (count % self.coding_step_interval_count == 0):
                    print(log)
            elif isinstance(log, PredictionModelTrainingProgressStep):
                self.training_progress_count += 1
                count = self.training_progress_count
                if log.total_steps is not None:
                    log.message = f"{log.base_message} ({count}/{log.total_steps})"
                else:
                    log.message = f"{log.base_message} ({count})"
                if self.record_progress:
                    self.logs.append(log)
                if self.display_progress and (count % self.training_step_interval_count == 0):
                    print(log)

    def save(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            for log in self.logs:
                file.write(str(log) + "\n")
