import argparse
import logging
import logging.config
from pathlib import Path
from src.definitions import (
    ANIMAL_CLASSIFIER_NAME,
    CAT_BIN_CLASSIFIER_NAME,
    GRAD_CHECK_FOLDER,
    LOGGING_CONFIG_PATH,
)
from src.models.animal_classifier.gradient_check_animal_classifier import (
    gradient_check_animal_classifier,
)
from src.models.cat_bin_classifier.gradient_check_cat_bin_classifier import (
    gradient_check_cat_bin_classifier,
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
        gradient_check_animal_classifier(GRAD_CHECK_FOLDER, logger)
    elif model == CAT_BIN_CLASSIFIER_NAME:
        gradient_check_cat_bin_classifier(GRAD_CHECK_FOLDER, logger)
    else:
        raise ValueError("Unknown model name " + model)
