import os
import random
import struct
import numpy as np
import tensorflow as tf
from typing import Optional, Any

from keras import mixed_precision

# It is preferable to import specific symbols rather than using wildcard imports.
from .validators import validate_type, validate_file_exists  
from .prediction_models import TfPredictionModel
from .preprocessors import BasePreprocessor, BytePreprocessor, get_preprocessor      
from .coders import CoderBase, get_coder         
from .keras_models import get_keras_model       
from .logger import Logger            

class CompressedModel:
    """Represents a compressed file."""

    def __init__(
        self,
        preprocessor_code: int,
        version: int,
        preprocessor_header_size: int,
        preprocessor_header: bytes,
        keras_code: int,
        coder_code: int,
        data: bytes,
        original_file_name: Optional[str] = None,
    ) -> None:
        validate_type(preprocessor_code, "Preprocessor code", int)
        validate_type(version, "Version", int)
        validate_type(preprocessor_header_size, "Preprocessor header size", int)
        validate_type(preprocessor_header, "Preprocessor header", bytes)
        validate_type(keras_code, "Keras code", int)
        validate_type(coder_code, "Coder code", int)
        validate_type(data, "Data", bytes)
        if original_file_name is not None:
            validate_type(original_file_name, "Original file name", str)

        preprocessor = get_preprocessor(preprocessor_code)

        if version != 1:
            raise ValueError("Version not supported")
        if preprocessor_header_size < 0:
            raise ValueError("Preprocessor header size must be non-negative")
        if len(preprocessor_header) > preprocessor_header_size:
            raise ValueError("Length of preprocessor header must be less than or equal to preprocessor header size")

        coder = get_coder(coder_code)

        self.original_file_name = original_file_name
        self.preprocessor_code = preprocessor_code
        self.version = version
        self.preprocessor_header_size = preprocessor_header_size
        self.preprocessor_header = preprocessor_header
        self.keras_code = keras_code
        self.coder_code = coder_code
        self.data = data

    @staticmethod
    def serialize(model: 'CompressedModel') -> bytes:
        """
        Serialize a CompressedModel instance into bytes.

        The format:
          - preprocessor_code (4 bytes, unsigned int)
          - version (4 bytes, unsigned int)
          - preprocessor_header_size (4 bytes, unsigned int)
          - actual header length (4 bytes, unsigned int)
          - preprocessor_header (variable length, actual header length bytes)
          - keras_code (4 bytes, unsigned int)
          - coder_code (4 bytes, unsigned int)
          - data length (4 bytes, unsigned int)
          - data (variable length)
          - original_file_name length (4 bytes, unsigned int; 0 if None)
          - original_file_name (UTF-8 encoded, if present)
        """
        header_actual_len = len(model.preprocessor_header)
        file_name_bytes = (
            model.original_file_name.encode("utf-8") if model.original_file_name is not None else b""
        )
        file_name_length = len(file_name_bytes)

        serialized = struct.pack(
            "IIII",
            model.preprocessor_code,
            model.version,
            model.preprocessor_header_size,
            header_actual_len,
        )
        serialized += model.preprocessor_header
        serialized += struct.pack("II", model.keras_code, model.coder_code)
        serialized += struct.pack("I", len(model.data))
        serialized += model.data
        serialized += struct.pack("I", file_name_length)
        serialized += file_name_bytes

        return serialized

    @staticmethod
    def deserialize(serialized: bytes) -> 'CompressedModel':
        """
        Deserialize bytes into a CompressedModel instance.
        The byte structure is expected to be the same as produced by serialize().
        """
        if len(serialized) < 16:
            raise ValueError("Serialized data is too short")
        preprocessor_code, version, preprocessor_header_size, header_actual_len = struct.unpack("IIII", serialized[:16])
        offset = 16

        preprocessor_header = serialized[offset : offset + header_actual_len]
        offset += header_actual_len

        if len(serialized) < offset + 4:
            raise ValueError("Serialized data is incomplete for keras_code")
        keras_code, = struct.unpack("I", serialized[offset : offset + 4])
        offset += 4

        if len(serialized) < offset + 4:
            raise ValueError("Serialized data is incomplete for coder_code")
        coder_code, = struct.unpack("I", serialized[offset : offset + 4])
        offset += 4

        if len(serialized) < offset + 4:
            raise ValueError("Serialized data is incomplete for data length")
        data_length, = struct.unpack("I", serialized[offset : offset + 4])
        offset += 4

        data = serialized[offset : offset + data_length]
        offset += data_length

        if len(serialized) < offset + 4:
            raise ValueError("Serialized data is incomplete for file name length")
        file_name_length, = struct.unpack("I", serialized[offset : offset + 4])
        offset += 4
        if file_name_length > 0:
            original_file_name = serialized[offset : offset + file_name_length].decode("utf-8")
        else:
            original_file_name = None

        return CompressedModel(
            preprocessor_code, version, preprocessor_header_size, preprocessor_header, keras_code, coder_code, data, original_file_name
        )


class CompressedModelFile:
    """Provides methods to write and read a CompressedModel instance to/from a file."""

    @staticmethod
    def write_to_file(model: CompressedModel, file_path: str) -> None:
        """
        Serialize the model and write it as binary data to the given file.

        Args:
            model (CompressedModel): The compressed model to write.
            file_path (str): The path to the output file.
        """
        serialized_data = CompressedModel.serialize(model)
        with open(file_path, "wb") as file:
            file.write(serialized_data)

    @staticmethod
    def read_from_file(file_path: str) -> CompressedModel:
        """
        Read binary data from the given file and deserialize it into a CompressedModel instance.

        Args:
            file_path (str): The path to the compressed file.

        Returns:
            CompressedModel: The deserialized compressed model.
        """
        with open(file_path, "rb") as file:
            serialized_data = file.read()
        return CompressedModel.deserialize(serialized_data)


def enable_determinism(seed: int) -> None:
    """
    Initialize random seeds for determinism.

    Args:
        seed (int): The seed value.
    """
    if seed is None:
        raise ValueError("Seed cannot be None")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    mixed_precision.set_global_policy("mixed_float16")


class TfCodec:
    def compress(
        self,
        data: bytes,
        preprocessor: Any,
        keras_model_code: int,
        coder: Any,
        pre_trained_model_path: Optional[str] = None,
        logger: Optional[Any] = None,
    ) -> CompressedModel:
        """
        Compress the input data.

        Args:
            data (bytes): The data to compress.
            preprocessor: An instance of BasePreprocessor.
            keras_model_code (int): Code identifying the Keras model.
            coder: An instance of CoderBase.
            pre_trained_model_path (Optional[str]): Path to pre-trained model weights, if any.
            logger: Logger instance for logging.

        Returns:
            CompressedModel: The resulting compressed model.
        """
        validate_type(data, "Data", bytes)
        if not isinstance(preprocessor, BasePreprocessor):
            raise ValueError("Preprocessor must be an instance of BasePreprocessor")
        validate_type(keras_model_code, "Keras model code", int)
        if not isinstance(coder, CoderBase):
            raise ValueError("Coder must be an instance of CoderBase")

        enable_determinism(42)

        prpr_code = preprocessor.code
        syms, dictionary = preprocessor.convert_to_symbols(data)
        prpr_header = preprocessor.encode_dictionary_for_header(dictionary)
        keras_model = get_keras_model(keras_model_code, dictionary.get_size())

        if pre_trained_model_path is not None:
            predictor = TfPredictionModel(dictionary, keras_model, model_weights_path=pre_trained_model_path, logger=logger)
        else:
            predictor = TfPredictionModel(dictionary, keras_model, logger=logger)
        encoded_data = coder.encode(syms, predictor)
        compressed_model = CompressedModel(
            prpr_code,
            1,
            preprocessor.header_size,
            prpr_header,
            keras_model_code,
            coder.coder_code,
            encoded_data,
        )
        return compressed_model

    def decompress(
        self,
        compressed_model: CompressedModel,
        pre_trained_model_path: Optional[str] = None,
        logger: Optional[Logger] = None,
    ) -> bytes:
        """
        Decompress the encoded data.

        Args:
            compressed_model (CompressedModel): The compressed model.
            pre_trained_model_path (Optional[str]): Path to pre-trained model weights, if any.
            logger: Logger instance for logging.

        Returns:
            bytes: The decompressed data.
        """
        if not isinstance(compressed_model, CompressedModel):
            raise ValueError("Input must be a CompressedModel instance")
        if compressed_model.version != 1:
            raise ValueError("Version not supported")

        preprocessor = get_preprocessor(compressed_model.preprocessor_code, logger=logger)
        if preprocessor is None:
            raise ValueError("Preprocessor not supported")

        dictionary = preprocessor.construct_dictionary_from_header(compressed_model.preprocessor_header)
        if dictionary is None:
            raise ValueError("Dictionary construction failed")

        keras_model = get_keras_model(compressed_model.keras_code, dictionary.get_size())
        if keras_model is None:
            raise ValueError("Keras model not supported")

        coder = get_coder(compressed_model.coder_code, logger=logger)
        if coder is None:
            raise ValueError("Coder not supported")

        enable_determinism(42)

        predictor = TfPredictionModel(dictionary, keras_model, model_weights_path=pre_trained_model_path, logger=logger)
        syms = coder.decode(compressed_model.data, dictionary, predictor)
        return preprocessor.convert_from_symbols(syms)


class TfCodecFile(TfCodec):
    def compress(
        self,
        input_path: str,
        output_path: str,
        preprocessor: Any,
        keras_model_code: int,
        coder: Any,
        pre_trained_model_path: Optional[str] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Compress the input file and write the compressed model to an output file.

        Args:
            input_path (str): Path to the input file.
            output_path (str): Path to the output file.
            preprocessor: An instance of BasePreprocessor.
            keras_model_code (int): Code identifying the Keras model.
            coder: An instance of CoderBase.
            pre_trained_model_path (Optional[str]): Path to pre-trained model weights, if any.
            logger: Logger instance for logging.
        """
        validate_type(input_path, "Input path", str)
        validate_type(output_path, "Output path", str)
        validate_file_exists(input_path)

        with open(input_path, "rb") as file:
            data = file.read()

        compressed_model = super().compress(data, preprocessor, keras_model_code, coder, pre_trained_model_path, logger)
        CompressedModelFile.write_to_file(compressed_model, output_path)

    def decompress(
        self,
        compressed_file_path: str,
        output_file_path: str,
        pre_trained_model_path: Optional[str] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Decompress the input file and write the decompressed data to an output file.

        Args:
            compressed_file_path (str): Path to the compressed file.
            output_file_path (str): Path to the output file.
            pre_trained_model_path (Optional[str]): Path to pre-trained model weights, if any.
            logger: Logger instance for logging.
        """
        validate_type(compressed_file_path, "Compressed file path", str)
        validate_type(output_file_path, "Output file path", str)
        validate_file_exists(compressed_file_path)

        compressed_model = CompressedModelFile.read_from_file(compressed_file_path)
        data = super().decompress(compressed_model, pre_trained_model_path, logger)
        with open(output_file_path, "wb") as file:
            file.write(data)


class TfCodecByte(TfCodec):
    def compress(
        self,
        data: bytes,
        keras_model_code: int,
        coder_code: int,
        pre_trained_model_path: Optional[str] = None,
        logger: Optional[Logger] = None,
    ) -> CompressedModel:
        """
        Compress the input data using a byte preprocessor.

        Args:
            data (bytes): The data to compress.
            keras_model_code (int): Code identifying the Keras model.
            coder_code (int): Code identifying the coder.
            pre_trained_model_path (Optional[str]): Path to pre-trained model weights, if any.
            logger: Logger instance for logging.

        Returns:
            CompressedModel: The resulting compressed model.
        """
        validate_type(data, "Data", bytes)
        validate_type(keras_model_code, "Keras model code", int)
        validate_type(coder_code, "Coder code", int)

        coder = get_coder(coder_code, logger=logger)
        return super().compress(data, BytePreprocessor(), keras_model_code, coder, pre_trained_model_path, logger)


class TfCodecByteFile(TfCodecFile):
    def compress(
        self,
        input_path: str,
        output_path: str,
        keras_model_code: int,
        coder_code: int,
        pre_trained_model_path: Optional[str],
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Compress the input file using a byte preprocessor and write the result to an output file.

        Args:
            input_path (str): Path to the input file.
            output_path (str): Path to the output file.
            keras_model_code (int): Code identifying the Keras model.
            coder_code (int): Code identifying the coder.
            pre_trained_model_path (Optional[str]): Path to pre-trained model weights.
            logger: Logger instance for logging.
        """
        validate_type(input_path, "Input path", str)
        validate_type(output_path, "Output path", str)
        validate_file_exists(input_path)

        coder = get_coder(coder_code, logger=logger)
        super().compress(input_path, output_path, BytePreprocessor(logger), keras_model_code, coder, pre_trained_model_path, logger)


class TfCodecByteArithmetic(TfCodecByte):
    def compress(
        self,
        data: bytes,
        keras_model_code: int,
        offline_learning: bool = True,
        pre_trained_model_path: Optional[str] = None,
        logger: Optional[Logger] = None,
    ) -> CompressedModel:
        """
        Compress the input data using arithmetic coding with a byte preprocessor.

        Args:
            data (bytes): The data to compress.
            keras_model_code (int): Code identifying the Keras model.
            offline_learning (bool): Whether to use deep arithmetic coding.
            pre_trained_model_path (Optional[str]): Path to pre-trained model weights, if any.
            logger: Logger instance for logging.

        Returns:
            CompressedModel: The resulting compressed model.
        """
        coder_code = 2 if offline_learning else 1
        return super().compress(data, keras_model_code, coder_code, pre_trained_model_path, logger)


class TfCodecByteArithmeticFile(TfCodecByteFile):
    def compress(
        self,
        input_path: str,
        output_path: str,
        keras_model_code: int,
        use_deep: bool = True,
        pre_trained_model_path: Optional[str] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Compress the input file using arithmetic coding with a byte preprocessor and write the result to an output file.

        Args:
            input_path (str): Path to the input file.
            output_path (str): Path to the output file.
            keras_model_code (int): Code identifying the Keras model.
            use_deep (bool): Whether to use deep arithmetic coding.
            pre_trained_model_path (Optional[str]): Path to pre-trained model weights, if any.
            logger: Logger instance for logging.
        """
        coder_code = 2 if use_deep else 1
        super().compress(input_path, output_path, keras_model_code, coder_code, pre_trained_model_path, logger)
