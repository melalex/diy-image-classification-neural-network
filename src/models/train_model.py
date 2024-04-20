import argparse
from pathlib import Path

import logging
import logging.config

import numpy as np

from src.definitions import (
    ANIMAL_CLASSIFIER_BASE_FILE_NAME,
    ANIMAL_CLASSIFIER_NAME,
    CAT_BIN_CLASSIFIER_BASE_FILE_NAME,
    CAT_BIN_CLASSIFIER_NAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LOGGING_CONFIG_PATH,
    MODELS_FOLDER,
    TRAIN_DATA_FOLDER,
)
from src.models.neural_network_def import (
    ANIMAL_HYPER_PARAMS,
    CAT_BIN_HYPER_PARAMS,
    animal_classifier,
    cat_bin_classifier,
)
from src.util.image import read_all_images_and_predictions
from src.util.persist_nn import create_model_path, write_model


def train_animal_classification_model(dataset: Path, logger: logging.Logger):
    model_path = create_model_path(
        MODELS_FOLDER, ANIMAL_CLASSIFIER_BASE_FILE_NAME, ANIMAL_HYPER_PARAMS
    )

    if model_path.is_file():
        logger.info(
            "Model file is present at [ %s ]. Skipping training ...", model_path
        )
        return

    x, y = read_all_images_and_predictions(dataset, IMAGE_WIDTH, IMAGE_HEIGHT)
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1

    logger.info("Loaded [ %s ] images", x.shape[1])

    model = animal_classifier()

    model.train(x, y_one_hot)

    logger.info("Saving trained model to [ %s ]", model_path)

    write_model(model, model_path)


def train_cat_bin_classification_model(dataset: Path, logger: logging.Logger):
    model_path = create_model_path(
        MODELS_FOLDER, CAT_BIN_CLASSIFIER_BASE_FILE_NAME, CAT_BIN_HYPER_PARAMS
    )

    if model_path.is_file():
        logger.info(
            "Model file is present at [ %s ]. Skipping training ...", model_path
        )
        return

    x, y = read_all_images_and_predictions(dataset, IMAGE_WIDTH, IMAGE_HEIGHT)

    logger.info("Loaded [ %s ] images", x.shape[1])

    model = cat_bin_classifier()

    model.train(x, y)

    logger.info("Saving trained model to [ %s ]", model_path)

    write_model(model, model_path)


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
        train_animal_classification_model(TRAIN_DATA_FOLDER, logger)
    elif model == CAT_BIN_CLASSIFIER_NAME:
        train_cat_bin_classification_model(TRAIN_DATA_FOLDER, logger)
    else:
        raise ValueError("Unknown model name " + model)
