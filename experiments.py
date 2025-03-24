#experiment.py
import time
import os

from nncodec.codec import *
from nncodec.performence_display import *
from nncodec.trainers import *
from nncodec.prediction_models import *
from nncodec.models import *
from nncodec.keras_models import *
from nncodec.logger import *
from nncodec.preprocessors import *


class TfByteArithmeticExperiment:
    def __init__(self, name: str, input_file_path, experiment_root_folder_path, keras_model_code, use_deep_learning = False, prelearning_folder = None):
        
        
        self.name = name
        
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"File {input_file_path} not found.")
        if not os.access(input_file_path, os.R_OK):
            raise PermissionError(f"File {input_file_path} is not readable.")
        
        self.input_file_path = input_file_path
        
        self.experiment_folder_path = os.path.join(experiment_root_folder_path, name)
        if not os.path.exists(self.experiment_folder_path):
            os.makedirs(self.experiment_folder_path)
        
        input_file_name = os.path.basename(input_file_path)
        self.compressed_file_path = os.path.join(self.experiment_folder_path, f"{input_file_name}.nncodec")
        self.decompressed_file_path = os.path.join(self.experiment_folder_path, f"{input_file_name}_decompressed")
    
        self.prelearning_folder = prelearning_folder        
        
        self.compression_logger = Logger()
        self.decompression_logger = Logger()
        self.codec = TfCodecByteArithmeticFile()
        self.keras_model_code = keras_model_code
        self.use_deep_learning = use_deep_learning
        
        
    def run(self):
        self.input_file_size = os.path.getsize(self.input_file_path)
        
        self.model_weights_path = os.path.join(self.experiment_folder_path, f"{self.name}_model.weights.h5")
        if self.prelearning_folder is not None:
            self.training_logger = Logger()
            preporcessor = BytePreprocessor(self.training_logger)
            with open(self.input_file_path, 'rb') as f:
                data = f.read()
            _, dictionary = preporcessor.convert_to_symbols(data)
            keras_model = get_keras_model(self.keras_model_code, dictionary.get_size())
            predictor = TfPredictionModel(dictionary, keras_model, logger=self.training_logger)
            trainer = TfTrainer()
            for file_name in os.listdir(self.prelearning_folder):
                with open(os.path.join(self.prelearning_folder, file_name), 'rb') as f:
                    data = f.read()
                symbols, f_dictionary = preporcessor.convert_to_symbols(data)
                if dictionary != f_dictionary:
                    continue
                trainer.train(predictor, symbols, self.training_logger)
            predictor.save_model(self.model_weights_path)
        else:
            self.model_weights_path = None            
        
        self.compression_start_time = time.time()
        self.codec.compress(self.input_file_path, self.compressed_file_path, self.keras_model_code, self.use_deep_learning, self.model_weights_path, self.compression_logger)
        self.compression_end_time = time.time()
        
        self.decompression_start_time = time.time()
        self.codec.decompress(self.compressed_file_path, self.decompressed_file_path, self.model_weights_path, self.decompression_logger)
        self.decompression_end_time = time.time()
        
        self.compressed_file_size = os.path.getsize(self.compressed_file_path)
        self.decompressed_file_size = os.path.getsize(self.decompressed_file_path)
        
        self.compression_ratio = self.input_file_size / self.compressed_file_size
        
    def save_report_in_text(self, file_path: str):
        if not os.path.exists(os.path.dirname(file_path)):
            raise FileNotFoundError(f"Folder {os.path.dirname(file_path)} not found.")
        if not os.access(os.path.dirname(file_path), os.W_OK):
            raise PermissionError(f"Folder {os.path.dirname(file_path)} is not writable.")
        
        if not hasattr(self, 'compression_start_time'):
            raise RuntimeError("The experiment has not been run yet.")
        
        with open(file_path, 'w') as f:
            f.write(f"Experiment name: {self.name}\n")
            f.write(f"Input file size: {self.input_file_size}\n")
            f.write(f"Compression time: {self.compression_end_time - self.compression_start_time}\n")
            f.write(f"Decompression time: {self.decompression_end_time - self.decompression_start_time}\n")
            f.write(f"Compressed file size: {self.compressed_file_size}\n")
            f.write(f"Decompressed file size: {self.decompressed_file_size}\n")
            f.write(f"Compression ratio: {self.compression_ratio}\n")
            
    def display_graphs(self):
        if not hasattr(self, 'compression_logger'):
            raise RuntimeError("The experiment has not been run yet.")
        
        display = PerformanceDisplay(self.compression_logger.logs)
        display.plot_coding_log(os.path.join(self.experiment_folder_path, f"{self.name}_coding_log.png"))
        display.plot_prediction_model_training_log(os.path.join(self.experiment_folder_path, f"{self.name}_prediction_model_training_log.png"))
        display.plot_encoded_symbol_probability(os.path.join(self.experiment_folder_path, f"{self.name}_encoded_symbol_probability.png"))
        
        
        


if __name__ == '__main__':
    experiments_output_path = 'experiments_out'
    input_path = 'experiments_data/enwiks/enwik4'
    
    #test experiment:
    # experiment_name = f"experiment_test_{time.strftime('%Y%m%d_%H%M%S')}"
    # input_path = 'experiments_data/enwiks_trianing_folder/enwik4'
    # experiment_training_folder = os.path.join("experiments_data", "enwiks_trianing_folder")
    # experiment = TfByteArithmeticExperiment(experiment_name, input_path, experiments_output_path, 0, False, experiment_training_folder)
    # experiment.run()
    # experiment.save_report_in_text(os.path.join(experiments_output_path, experiment_name, f"{experiment_name}.txt"))
    # experiment.display_graphs()

    #lstm_experiment (online learning)
    experiment_name = f"experiment_lstm_online_{time.strftime('%Y%m%d_%H%M%S')}"
    experiment = TfByteArithmeticExperiment(experiment_name, input_path, experiments_output_path, 1, False)
    experiment.run()
    experiment.save_report_in_text(os.path.join(experiments_output_path, experiment_name, f"{experiment_name}.txt"))
    
    #lstm_experiment (deep learning)
    experiment_name = f"experiment_lstm_deep_{time.strftime('%Y%m%d_%H%M%S')}"
    experiment = TfByteArithmeticExperiment(experiment_name, input_path, experiments_output_path, 1, True)
    experiment.run()
    experiment.save_report_in_text(os.path.join(experiments_output_path, experiment_name, f"{experiment_name}.txt"))
    
    #gru_experiment (online learning)
    experiment_name = f"experiment_gru_online_{time.strftime('%Y%m%d_%H%M%S')}"
    experiment = TfByteArithmeticExperiment(experiment_name, input_path, experiments_output_path, 2, False)
    experiment.run()
    experiment.save_report_in_text(os.path.join(experiments_output_path, f"{experiment_name}.txt"))
    
    # #training experiments:
    # input_path = 'experiments_data/wikipedia_large_pages_txt/List of subnational entities by Human Development Index - Wikipedia.txt'
    # # lstm_experiment (online learning)
    # experiment_name = f"experiment_lstm_online_no_training_{time.strftime('%Y%m%d_%H%M%S')}"
    # experiment = TfByteArithmeticExperiment(experiment_name, input_path, experiments_output_path, 1, False)
    # experiment.run()
    # experiment.save_report_in_text(os.path.join(experiments_output_path, f"{experiment_name}.txt"))
    
    # lstm_experiment (pre learning)
    # experiment_name = f"experiment_lstm_prelearning_{time.strftime('%Y%m%d_%H%M%S')}"
    # experiment_training_folder = os.path.join("experiments_data", "wikipedia_large_pages_txt")
    # experiment = TfByteArithmeticExperiment(experiment_name, input_path, experiments_output_path, 1, False, experiment_training_folder)
    # experiment.run()
    # experiment.save_report_in_text(os.path.join(experiments_output_path, f"{experiment_name}.txt"))
    
    