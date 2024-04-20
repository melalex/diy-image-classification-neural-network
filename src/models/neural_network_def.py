from lib.activation_function import (
    RELU_ACTIVATION_FUN,
    SIGMOID_ACTIVATION_FUN,
    SOFT_MAX_ACTIVATION_FUN,
)
from lib.cost_function import (
    BINARY_CLASSIFICATION_COST_FUN,
    MULTI_CLASS_CLASSIFICATION_COST_FUN,
)
from lib.neural_network import HyperParams, NeuralNetwork
from lib.progress_tracker import LoggingProgressTracker
from lib.stop_condition import IterCountStopCondition
from src.definitions import IMAGE_HEIGHT, IMAGE_VECTOR_SHAPE, IMAGE_WIDTH, LOG_PERIOD


ANIMAL_HYPER_PARAMS = HyperParams(
    learning_rate=0.005,
    stop_condition=IterCountStopCondition(1000),
    init_weight_scale=0.01,
)

CAT_BIN_HYPER_PARAMS = HyperParams(
    learning_rate=0.005,
    stop_condition=IterCountStopCondition(1000),
    init_weight_scale=0.01,
)


def animal_classifier():
    return (
        NeuralNetwork.builder()
        .with_hyper_params(ANIMAL_HYPER_PARAMS)
        .with_feature_count(IMAGE_VECTOR_SHAPE)
        .with_cost_fun(MULTI_CLASS_CLASSIFICATION_COST_FUN)
        # .with_layer(180, RELU_ACTIVATION_FUN)
        # .with_layer(180, RELU_ACTIVATION_FUN)
        .with_layer(1, SOFT_MAX_ACTIVATION_FUN)
        .with_progress_tracker(LoggingProgressTracker(LOG_PERIOD))
        .build()
    )


def cat_bin_classifier():
    return (
        NeuralNetwork.builder()
        .with_hyper_params(CAT_BIN_HYPER_PARAMS)
        .with_feature_count(IMAGE_VECTOR_SHAPE)
        .with_cost_fun(BINARY_CLASSIFICATION_COST_FUN)
        # .with_layer(180, RELU_ACTIVATION_FUN)
        # .with_layer(180, RELU_ACTIVATION_FUN)
        .with_layer(1, SIGMOID_ACTIVATION_FUN)
        .with_progress_tracker(LoggingProgressTracker(LOG_PERIOD))
        .build()
    )
