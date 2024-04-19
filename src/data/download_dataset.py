import kaggle
import logging
import logging.config

from src.definitions import (
    CATS_VS_DOGS_DATASET_NAME,
    CATS_VS_DOGS_DATASET_OWNER,
    LOGGING_CONFIG_PATH,
    RAW_DATA_FOLDER,
    SAMPLE_DATASET_NAME,
    SAMPLE_DATASET_OWNER,
)
from src.util.dataset import download_dataset


def download_all_datasets(logger: logging.Logger):
    kaggle.api.authenticate()
    download_dataset(
        CATS_VS_DOGS_DATASET_OWNER, CATS_VS_DOGS_DATASET_NAME, RAW_DATA_FOLDER, logger
    )
    download_dataset(SAMPLE_DATASET_OWNER, SAMPLE_DATASET_NAME, RAW_DATA_FOLDER, logger)


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    download_all_datasets(logging.getLogger(__name__))
