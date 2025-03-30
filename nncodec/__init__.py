"""
NNCodec: A Python library for lossless neural network-based data compression and decompression.
"""

from .codecs import (
    CompressedModel,
    CompressedModelFile,
    TfCodec,
    TfCodecFile,
    TfCodecByte,
    TfCodecByteFile,
    TfCodecByteArithmetic,
    TfCodecByteArithmeticFile,
)

from .coders import (
    CoderBase,
    ArithmeticCoderSettings,
    BitOutputStream,
    BitInputStream,
    ArithmeticCodec,
    ArithmeticCoderOnline,
    ArithmeticCoderOffline,
    get_coder,
)

from .models import (
    Symbol,
    SymbolFrequency,
    Dictionary,
)

from .keras_models import (
    TFKerasModelBase,
    LstmKerasModel,
    GruKerasModel,
    LstmKerasModelLight,
    TestingKerasModel,
    get_keras_model,
)

from .prediction_models import (
    BasePredictionModel,
    TestingPredictionModel,
    TfPredictionModel,
)

from .preprocessors import (
    BasePreprocessor,
    AsciiCharPreprocessor,
    BytePreprocessor,
    get_preprocessor,
)

from .trainers import (
    BaseTrainer,
    TfTrainer,
)

from .settings import SEED

from .logger import (
    Logger,
    Log,
    LogLevel,
    CodingLog,
    EncodedSymbolProbability,
    PredictionModelTrainingLog,
    PreprocessingProgressStep,
    CodingProgressStep,
    PredictionModelTrainingProgressStep,
)

# Validators
from .validators import *

__all__ = [
    
    "CompressedModel",
    "CompressedModelFile",
    "TfCodec",
    "TfCodecFile",
    "TfCodecByte",
    "TfCodecByteFile",
    "TfCodecByteArithmetic",
    "TfCodecByteArithmeticFile",
    
    "CoderBase",
    "ArithmeticCoderSettings",
    "BitOutputStream",
    "BitInputStream",
    "ArithmeticCodec",
    "ArithmeticCoderOnline",
    "ArithmeticCoderOffline",
    "get_coder",
    
    "CompressionResult",
    "Symbol",
    "SymbolFrequency",
    "Dictionary",
    
    "TFKerasModelBase",
    "LstmKerasModel",
    "GruKerasModel",
    "LstmKerasModelLight",
    "TestingKerasModel",
    "get_keras_model",
    
    "BasePredictionModel",
    "TestingPredictionModel",
    "TfPredictionModel",
    
    "BasePreprocessor",
    "AsciiCharPreprocessor",
    "BytePreprocessor",
    "get_preprocessor",
    
    "BaseTrainer",
    "TfTrainer",
]
