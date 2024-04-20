import logging
from pathlib import Path

import numpy as np

from lib.neural_network_ext import NeuralNetworkExt
from lib.persist_neural_network import read_model
from src.definitions import (
    ANIMAL_CLASSIFIER_BASE_FILE_NAME,
    LOGGING_CONFIG_PATH,
    MODELS_FOLDER,
    TEST_DATA_FOLDER,
    TRAIN_DATA_FOLDER,
)
from src.models.animal_classifier.animal_classifier_def import ANIMAL_HYPER_PARAMS
from src.models.animal_classifier.animal_classifier_util import (
    read_animal_classifier_data_and_labels,
)


def test_animal_classifier(logger: logging.Logger):
    model = read_model(
        MODELS_FOLDER, ANIMAL_CLASSIFIER_BASE_FILE_NAME, ANIMAL_HYPER_PARAMS
    )
    train_result = test_animal_classifier_with(model, TRAIN_DATA_FOLDER)
    test_result = test_animal_classifier_with(model, TEST_DATA_FOLDER)

    logger.info("Train accuracy: %s %%", train_result)
    logger.info("Test accuracy: %s %%", test_result)


def test_animal_classifier_with(model: NeuralNetworkExt, path: Path) -> float:
    x, y = read_animal_classifier_data_and_labels(path)

    a = model.predict(x)

    return 100 - np.mean(np.abs(a - y)) * 100


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    test_animal_classifier(logging.getLogger(__name__))
