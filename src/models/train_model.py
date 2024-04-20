import argparse
from pathlib import Path

import logging
import logging.config

import numpy as np

from src.definitions import (
    ANIMAL_CLASSIFIER_NAME,
    CAT_BIN_CLASSIFIER_NAME,
    LOGGING_CONFIG_PATH,
    TRAIN_DATA_FOLDER,
)
from src.models.animal_classifier.train_animal_classifier import (
    train_animal_classification_and_persist,
)
from src.models.cat_bin_classifier.train_cat_bin_classifier import (
    train_cat_bin_classification_and_persist,
)

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
        train_animal_classification_and_persist(TRAIN_DATA_FOLDER, logger)
    elif model == CAT_BIN_CLASSIFIER_NAME:
        train_cat_bin_classification_and_persist(TRAIN_DATA_FOLDER, logger)
    else:
        raise ValueError("Unknown model name " + model)
