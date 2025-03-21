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
