#experiment.py
import time

from .codec import *


class TfByteArithmeticExperiment:
    def __init__(self, name: str, input_file_path, experiment_root_folder_path, keras_model_code, use_deep_learning = False):
        
        self.name = name
        
        #validate that input file exists and can be read
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"File {input_file_path} not found.")
        if not os.access(input_file_path, os.R_OK):
            raise PermissionError(f"File {input_file_path} is not readable.")
        
        self.input_file_path = input_file_path
        
        #attempt to create experiment folder
        self.experiment_folder_path = os.path.join(experiment_root_folder_path, name)
        if not os.path.exists(self.experiment_folder_path):
            os.makedirs(self.experiment_folder_path)
        
        input_file_name = os.path.basename(input_file_path)
        self.compressed_file_path = os.path.join(self.experiment_folder_path, f"{input_file_name}.nncodec")
        self.decompressed_file_path = os.path.join(self.experiment_folder_path, f"{input_file_name}_decompressed")
        
        self.logger = Logger()
        self.codec = TfCodecByteArithmeticFile()
        self.keras_model_code = keras_model_code
        self.use_deep_learning = use_deep_learning
        
        
    def run(self):
        self.input_file_size = os.path.getsize(self.input_file_path)
        
        self.compression_start_time = time.time()
        self.codec.compress(self.input_file_path, self.compressed_file_path, self.keras_model_code, self.use_deep_learning, self.logger)
        self.compression_end_time = time.time()
        
        self.decompression_start_time = time.time()
        self.codec.decompress(self.compressed_file_path, self.decompressed_file_path, self.logger)
        self.decompression_end_time = time.time()
        
        self.compressed_file_size = os.path.getsize(self.compressed_file_path)
        self.decompressed_file_size = os.path.getsize(self.compressed_file_path + '_decompressed')
        
        self.compression_ratio = self.input_file_size / self.compressed_file_size
        
        
        