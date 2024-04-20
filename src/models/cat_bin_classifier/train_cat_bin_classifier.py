import logging
import logging.config
from pathlib import Path

from lib.persist_neural_network import create_model_path, write_model
from src.definitions import (
    CAT_BIN_CLASSIFIER_BASE_FILE_NAME,
    LOGGING_CONFIG_PATH,
    MODELS_FOLDER,
    TRAIN_DATA_FOLDER,
)
from src.models.cat_bin_classifier.cat_bin_classifier_def import (
    CAT_BIN_HYPER_PARAMS,
    cat_bin_classifier,
)
from src.models.cat_bin_classifier.cat_bin_classifier_util import (
    read_cat_bin_classifier_data_and_labels,
)


def train_cat_bin_classification(dataset: Path, logger: logging.Logger):
    x, y = read_cat_bin_classifier_data_and_labels(dataset)

    logger.info("Loaded [ %s ] images", x.shape[1])

    model = cat_bin_classifier(x.shape[0])

    model.train(x, y)

    return model


def train_cat_bin_classification_and_persist(dataset: Path, logger: logging.Logger):
    model_path = create_model_path(
        MODELS_FOLDER, CAT_BIN_CLASSIFIER_BASE_FILE_NAME, CAT_BIN_HYPER_PARAMS
    )

    if model_path.is_file():
        logger.info(
            "Model file is present at [ %s ]. Skipping training ...", model_path
        )
        return

    model = train_cat_bin_classification(dataset, logger)

    write_model(MODELS_FOLDER, CAT_BIN_CLASSIFIER_BASE_FILE_NAME, model)


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)

    train_cat_bin_classification_and_persist(
        TRAIN_DATA_FOLDER, logging.getLogger(__name__)
    )
