from lib.activation_function import RELU_ACTIVATION_FUN, SOFT_MAX_ACTIVATION_FUN
from lib.cost_function import MULTI_CLASS_CLASSIFICATION_COST_FUN
from lib.neural_network import HyperParams, NeuralNetwork
from lib.progress_tracker import LoggingProgressTracker
from src.definitions import LOG_PERIOD


def create_neural_network(features_count: int, learning_factor: float, iter_count: int):
    return (
        NeuralNetwork.builder()
        .with_iter_count(iter_count)
        .with_hyper_params(HyperParams(learning_factor))
        .with_feature_count(features_count)
        .with_cost_fun(MULTI_CLASS_CLASSIFICATION_COST_FUN)
        .with_layer(1, RELU_ACTIVATION_FUN)
        .with_layer(3, SOFT_MAX_ACTIVATION_FUN)
        .with_progress_tracker(LoggingProgressTracker(LOG_PERIOD))
        .build()
    )
