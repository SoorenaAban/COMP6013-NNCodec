#logger_test.py
#find a better way to this.

import io
import sys
import unittest
from nncodec.logger import Logger, Log, LogLevel, PreprocessingSymbolCreationLog

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.saved_stdout = sys.stdout
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output

    def tearDown(self):
        sys.stdout = self.saved_stdout

    def test_log_string(self):
        test_message = "Test log string"
        self.logger.log(test_message)
        
        self.assertEqual(len(self.logger.logs), 1)
        
        log_entry = self.logger.logs[0]
        self.assertEqual(log_entry.type_name, "General")
        self.assertEqual(log_entry.message, test_message)
        self.assertEqual(log_entry.level, LogLevel.INFO)
        
        printed_output = self.captured_output.getvalue()
        self.assertIn("General", printed_output)
        self.assertIn(test_message, printed_output)

    def test_log_log_instance(self):
        log_instance = PreprocessingSymbolCreationLog("A")
        self.logger.log(log_instance)
        
        self.assertEqual(len(self.logger.logs), 1)
        self.assertEqual(self.logger.logs[0], log_instance)
        
        printed_output = self.captured_output.getvalue()
        self.assertIn("Preprocessing_symbol_creation_log", printed_output)
        self.assertIn("Symbol: A", printed_output)

    def test_invalid_log(self):
        with self.assertRaises(ValueError):
            self.logger.log(123) 

    def test_warning_logging(self):
        warning_log = Log("WarningTest", LogLevel.WARNING, "This is a warning")
        self.logger.log(warning_log)
        self.assertEqual(len(self.logger.logs), 1)
        printed_output = self.captured_output.getvalue()
        self.assertIn("This is a warning", printed_output)

    def test_error_logging(self):
        error_log = Log("ErrorTest", LogLevel.ERROR, "This is an error")
        self.logger.log(error_log)
        self.assertEqual(len(self.logger.logs), 1)
        
        printed_output = self.captured_output.getvalue()
        self.assertIn("This is an error", printed_output)

if __name__ == '__main__':
    unittest.main()
