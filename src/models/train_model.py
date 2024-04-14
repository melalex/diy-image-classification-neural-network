from pathlib import Path

import numpy as np
import logging
import logging.config

from definitions import (
    ITERATION_COUNT,
    LEARNING_RATE,
    LOG_PERIOD,
    LOGGING_CONFIG_PATH,
    MODEL_PATH,
    TRAIN_DATA_FOLDER,
)
from lib.activation_function import RELU_ACTIVATION_FUN
from lib.cost_function import BINARY_CLASSIFICATION_COST_FUN
from lib.neural_network import HyperParams
from lib.neural_network_builder import NeuralNetworkBuilder
from lib.progress_tracker import LoggingProgressTracker
from src.util.image import read_all_images_and_predictions
from src.util.persist_nn import write_model


def train_model(
    dataset: Path, learning_factor: float, iter_count: int, logger: logging.Logger
):
    if MODEL_PATH.is_file():
        logger.info(
            "Model file is present at [ %s ]. Skipping training ...", MODEL_PATH
        )
        return

    x, y = read_all_images_and_predictions(dataset)

    nn = (
        NeuralNetworkBuilder()
        .with_iter_count(iter_count)
        .with_hyper_params(HyperParams(learning_factor))
        .with_feature_count(x.shape[0])
        .with_cost_fun(BINARY_CLASSIFICATION_COST_FUN)
        .with_layer(1, RELU_ACTIVATION_FUN)
        .with_progress_tracker(LoggingProgressTracker(LOG_PERIOD))
        .build()
    )

    nn.train(x, y)

    logger.info("Saving trained model to [ %s ]", MODEL_PATH)

    write_model(nn, MODEL_PATH)


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    train_model(
        TRAIN_DATA_FOLDER, LEARNING_RATE, ITERATION_COUNT, logging.getLogger(__name__)
    )
