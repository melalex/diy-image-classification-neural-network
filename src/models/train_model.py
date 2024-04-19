from pathlib import Path

import logging
import logging.config

from src.definitions import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ITERATION_COUNT,
    LEARNING_RATE,
    LOGGING_CONFIG_PATH,
    MODEL_PATH,
    TRAIN_DATA_FOLDER,
)
from src.models.neural_network_def import create_neural_network
from src.util.image import read_all_images_and_predictions
from src.util.persist_nn import write_model


def train_model(
    dataset: Path, learning_factor: float, iter_count: int, logger: logging.Logger
):
    x, y = read_all_images_and_predictions(dataset, IMAGE_WIDTH, IMAGE_HEIGHT)

    logger.info("Loaded [ %s ] images", x.shape[1])

    nn = create_neural_network(x.shape[0], learning_factor, iter_count)

    nn.train(x, y)

    return nn


def train_and_persist_model(
    dataset: Path,
    learning_factor: float,
    iter_count: int,
    model_path: Path,
    logger: logging.Logger,
):
    if model_path.is_file():
        logger.info(
            "Model file is present at [ %s ]. Skipping training ...", model_path
        )
        return

    model = train_model(dataset, learning_factor, iter_count, logger)

    logger.info("Saving trained model to [ %s ]", model_path)

    write_model(model, model_path)


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    train_and_persist_model(
        TRAIN_DATA_FOLDER,
        LEARNING_RATE,
        ITERATION_COUNT,
        MODEL_PATH,
        logging.getLogger(__name__),
    )
