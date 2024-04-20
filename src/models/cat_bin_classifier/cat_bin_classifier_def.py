import numpy as np
from lib.activation_function import SIGMOID_ACTIVATION_FUN
from lib.cost_function import BINARY_CLASSIFICATION_COST_FUN
from lib.neural_network import HyperParams
from lib.neural_network_ext import NeuralNetworkExt
from lib.progress_tracker import LoggingProgressTracker
from lib.stop_condition import IterCountStopCondition
from src.definitions import IMAGE_VECTOR_SHAPE, LOG_PERIOD


CAT_BIN_HYPER_PARAMS = HyperParams(
    learning_rate=0.005,
    beta=0.9,
    batch_size=512,
    stop_condition=IterCountStopCondition(1000),
    init_weight_scale=0.01,
)


def cat_bin_classifier(input_features_count: int) -> NeuralNetworkExt:
    return (
        NeuralNetworkExt.builder()
        .with_hyper_params(CAT_BIN_HYPER_PARAMS)
        .with_feature_count(input_features_count)
        .with_cost_fun(BINARY_CLASSIFICATION_COST_FUN)
        # .with_layer(180, RELU_ACTIVATION_FUN)
        # .with_layer(180, RELU_ACTIVATION_FUN)
        .with_layer(1, SIGMOID_ACTIVATION_FUN)
        .with_progress_tracker(LoggingProgressTracker(LOG_PERIOD))
        .build()
    )
