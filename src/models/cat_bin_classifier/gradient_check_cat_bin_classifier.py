import logging
from pathlib import Path

from src.definitions import GRAD_CHECK_FOLDER, LOGGING_CONFIG_PATH
from src.models.cat_bin_classifier.cat_bin_classifier_def import cat_bin_classifier
from src.models.cat_bin_classifier.cat_bin_classifier_util import (
    read_cat_bin_classifier_data_and_labels,
)


def gradient_check_cat_bin_classifier(dataset: Path, logger: logging.Logger) -> float:
    x, y = read_cat_bin_classifier_data_and_labels(dataset)

    logger.info("Loaded [ %s ] images", x.shape[1])

    nn = cat_bin_classifier(x.shape[0])

    gradient_check_diff = nn.gradient_check(x, y)

    logger.info("Gradient check result: " + str(gradient_check_diff))

    return gradient_check_diff


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    gradient_check_cat_bin_classifier(
        GRAD_CHECK_FOLDER,
        logging.getLogger(__name__),
    )
