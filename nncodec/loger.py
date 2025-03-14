#loger.py

from .models import LogMessage, LogType, ModuleStage, LogProgress

class Logger:
    def __init__(self):
        self.logs = []
        pass

    def log_symbol_encoded(self, symbol, encoded_size):
        self.logs.append(LogMessage(LogType.INFO, ))

    def get_logs(self):
        return self.logs

    def clear_logs(self):
        self.logs = []
        

MODULE_LOGGER = Logger()