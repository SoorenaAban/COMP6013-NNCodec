import time
import os

from nncodec.codec import *
from nncodec.logger import *
from nncodec.performence_display import *

enwik8_path = 'experiments_data/enwiks/enwik8'

#check if enwik8 file exists
if not os.path.exists(enwik8_path):
    raise FileNotFoundError(f"File {enwik8_path} not found.")

experiments_output_path = 'experiments_data/enwiks/experiments_output'
experiment_name = f"enwik8_experiment_{time.strftime('%Y%m%d_%H%M%S')}"
experiment_output_path = os.path.join(experiments_output_path, experiment_name)

compressed_enwik8_path_normal = os.path.join(experiment_output_path, 'enwik8_compressed_normal')
decompressed_enwik8_path_normal = os.path.join(experiment_output_path, 'enwik8_decompressed_normal')

compressed_enwik8_path_deep = os.path.join(experiment_output_path, 'enwik8_compressed_deep')
decompressed_enwik8_path_deep = os.path.join(experiment_output_path, 'enwik8_decompressed_deep')



#Create experiment folder if it doesn't exist
if not os.path.exists(experiment_output_path):
    os.makedirs(experiment_output_path)

logger = Logger()
nn_codec = TfCodecByteArithmeticFile()

enwik8_file_size = os.path.getsize(enwik8_path)

#normal arithmetic coding:
ari_comp_start_time = time.time()
nn_codec.compress(enwik8_path, compressed_enwik8_path_normal, 1, False, logger)
ari_comp_end_time = time.time()

ari_decomp_start_time = time.time()
nn_codec.decompress(compressed_enwik8_path_normal, compressed_enwik8_path_normal + '_decompressed', logger)
ari_decomp_end_time = time.time()

enwik8_file_size_compressed_normal = os.path.getsize(compressed_enwik8_path_normal)
enwik8_file_size_decompressed_normal = os.path.getsize(compressed_enwik8_path_normal + '_decompressed')
enwik8_compression_ratio_normal = enwik8_file_size / enwik8_file_size_compressed_normal

performence_display = PerformanceDisplay(logger.logs)

performence_display.plot_coding_log(os.path.join(experiment_output_path, 'coding_log_normal.png'))
performence_display.plot_prediction_model_training_log(os.path.join(experiment_output_path, 'prediction_model_training_log_normal.png'))
performence_display.plot_prediction_model_training_log(os.path.join(experiment_output_path, 'prediction_model_training_log_normal.png'))



