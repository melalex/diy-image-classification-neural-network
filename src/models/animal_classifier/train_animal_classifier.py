import logging
import logging.config
from pathlib import Path

from lib.persist_neural_network import create_model_path, write_model
from src.definitions import (
    ANIMAL_CLASSIFIER_BASE_FILE_NAME,
    LOGGING_CONFIG_PATH,
    MODELS_FOLDER,
    TRAIN_DATA_FOLDER,
)
from src.models.animal_classifier.animal_classifier_def import (
    animal_classifier,
    ANIMAL_HYPER_PARAMS,
)
from src.models.animal_classifier.animal_classifier_util import (
    read_animal_classifier_data_and_labels,
)


def train_animal_classification(dataset: Path, logger: logging.Logger):
    x, y = read_animal_classifier_data_and_labels(dataset)

    logger.info("Loaded [ %s ] images", x.shape[1])

    model = animal_classifier(x.shape[0])

    model.train(x, y)

    return model


def train_animal_classification_and_persist(dataset: Path, logger: logging.Logger):
    model_path = create_model_path(
        MODELS_FOLDER, ANIMAL_CLASSIFIER_BASE_FILE_NAME, ANIMAL_HYPER_PARAMS
    )

    if model_path.is_file():
        logger.info(
            "Model file is present at [ %s ]. Skipping training ...", model_path
        )
        return

    model = train_animal_classification(dataset, logger)

    write_model(MODELS_FOLDER, ANIMAL_CLASSIFIER_BASE_FILE_NAME, model)


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)

    train_animal_classification_and_persist(
        TRAIN_DATA_FOLDER, logging.getLogger(__name__)
    )
