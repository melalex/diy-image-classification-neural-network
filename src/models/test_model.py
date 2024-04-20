import argparse
import logging
import logging.config
from pathlib import Path

import numpy as np

from src.definitions import (
    ANIMAL_CLASSIFIER_BASE_FILE_NAME,
    ANIMAL_CLASSIFIER_NAME,
    CAT_BIN_CLASSIFIER_NAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LOGGING_CONFIG_PATH,
    MODELS_FOLDER,
    TEST_DATA_FOLDER,
    TRAIN_DATA_FOLDER,
)
from lib.neural_network import NeuralNetwork
from src.models.neural_network_def import ANIMAL_HYPER_PARAMS
from src.util.image import read_all_images_and_predictions
from src.util.persist_nn import create_model_path, read_model


def test_animal_classification_model(logger: logging.Logger):
    model = read_model(
        MODELS_FOLDER, ANIMAL_CLASSIFIER_BASE_FILE_NAME, ANIMAL_HYPER_PARAMS
    )
    x, y = read_all_images_and_predictions(dataset, IMAGE_WIDTH, IMAGE_HEIGHT)
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1

    train_result = test_model_with(model, TRAIN_DATA_FOLDER)
    test_result = test_model_with(model, TEST_DATA_FOLDER)

    logger.info("Train accuracy: %s %%", train_result)
    logger.info("Test accuracy: %s %%", test_result)


def test_model_with(model: NeuralNetwork, x: np.ndarray, y: np.ndarray) -> float:
    a = model.predict(x)

    return 100 - np.mean(np.abs(a - y)) * 100


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)

    parser = argparse.ArgumentParser(
        prog="diy-image-classification-neural-network",
    )

    parser.add_argument("model")

    args = parser.parse_args()

    model = args.model
    logger = logging.getLogger(__name__)

    if model == ANIMAL_CLASSIFIER_NAME:
        pass
    elif model == CAT_BIN_CLASSIFIER_NAME:
        pass
    else:
        raise ValueError("Unknown model name " + model)
