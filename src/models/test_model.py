import logging
import logging.config
from pathlib import Path

import numpy as np

from src.definitions import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LOGGING_CONFIG_PATH,
    MODEL_PATH,
    TEST_DATA_FOLDER,
    TRAIN_DATA_FOLDER,
)
from lib.neural_network import NeuralNetwork
from src.util.image import read_all_images_and_predictions
from src.util.persist_nn import read_model


def test_model(logger: logging.Logger):
    model = read_model(MODEL_PATH)
    train_result = test_model_with(model, TRAIN_DATA_FOLDER)
    test_result = test_model_with(model, TEST_DATA_FOLDER)

    logger.info("Train accuracy: %s %%", train_result)
    logger.info("Test accuracy: %s %%", test_result)


def test_model_with(model: NeuralNetwork, path: Path) -> float:
    x, y = read_all_images_and_predictions(path, IMAGE_WIDTH, IMAGE_HEIGHT)

    a = model.predict(x)

    return 100 - np.mean(np.abs(a - y)) * 100


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    test_model(logging.getLogger(__name__))
