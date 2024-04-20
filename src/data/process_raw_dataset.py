from pathlib import Path
from PIL import Image

import random
import logging
import logging.config

from src.definitions import (
    CATS_DATASET_CONTENT_PATH,
    CATS_VS_DOGS_DATASET_NAME,
    DOGS_DATASET_CONTENT_PATH,
    GRAD_CHECK_FOLDER,
    GRAD_CHECK_SAMPLES_COUNT,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LOG_PERIOD,
    LOGGING_CONFIG_PATH,
    RAW_DATA_FOLDER,
    SAMPLE_DATASET_CONTENT_PATH,
    SAMPLE_DATASET_NAME,
    SAMPLE_IMAGES_RATIO,
    TEST_DATA_FOLDER,
    TEST_TRAIN_RATIO,
    TRAIN_DATA_FOLDER,
    VALID_DATASET_FLAG_FILE,
)
from src.util.dataset import unzip_file


def process_raw_dataset(logger: logging.Logger):
    if VALID_DATASET_FLAG_FILE.is_file():
        logger.info("Dataset is already prepared. Skipping ...")
        return

    cats_vs_dogs_dataset_path = unzip_file(
        RAW_DATA_FOLDER, CATS_VS_DOGS_DATASET_NAME, logger
    )
    cats_dataset_path = cats_vs_dogs_dataset_path / CATS_DATASET_CONTENT_PATH
    dogs_dataset_path = cats_vs_dogs_dataset_path / DOGS_DATASET_CONTENT_PATH
    sample_dataset_path = (
        unzip_file(RAW_DATA_FOLDER, SAMPLE_DATASET_NAME, logger)
        / SAMPLE_DATASET_CONTENT_PATH
    )

    cat_images, dog_images, sample_images = list_images(
        cats_dataset_path, dogs_dataset_path, sample_dataset_path, logger
    )

    train_cat_images_count = int(len(cat_images) * TEST_TRAIN_RATIO)
    train_dog_images_count = int(len(dog_images) * TEST_TRAIN_RATIO)
    train_sample_images_count = int(len(sample_images) * TEST_TRAIN_RATIO)

    logger.info(
        "Preparing [ %s ] cat, [ %s ] dog, and [ %s ] images for train dataset",
        train_cat_images_count,
        train_dog_images_count,
        train_sample_images_count,
    )
    logger.info(
        "Preparing [ %s ] cat, [ %s ] dog, and [ %s ] images for test dataset",
        len(cat_images) - train_cat_images_count,
        len(dog_images) - train_dog_images_count,
        len(sample_images) - train_sample_images_count,
    )

    train_cat_images = cat_images[:train_cat_images_count]
    test_cat_images = cat_images[train_cat_images_count:]
    train_dog_images = dog_images[:train_dog_images_count]
    test_dog_images = dog_images[train_dog_images_count:]
    train_sample_images = sample_images[:train_sample_images_count]
    test_sample_images = sample_images[train_sample_images_count:]

    train_dataset = train_cat_images + train_dog_images + train_sample_images
    test_dataset = test_cat_images + test_dog_images + test_sample_images

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    write_to(train_dataset[:GRAD_CHECK_SAMPLES_COUNT], GRAD_CHECK_FOLDER, logger)
    write_to(train_dataset, TRAIN_DATA_FOLDER, logger)
    write_to(test_dataset, TEST_DATA_FOLDER, logger)

    VALID_DATASET_FLAG_FILE.touch()


def list_images(
    cats_dataset_path: Path,
    dogs_dataset_path: Path,
    sample_dataset_path: Path,
    logger: logging.Logger,
) -> tuple[list[Path], list[Path]]:
    cat_images = [it for it in cats_dataset_path.iterdir() if it.suffix == ".jpg"]
    dog_images = [it for it in dogs_dataset_path.iterdir() if it.suffix == ".jpg"]
    sample_images = [it for it in sample_dataset_path.iterdir() if it.suffix == ".jpg"]

    sample_images_count = len(sample_images)
    cat_images_count = sample_images_count // SAMPLE_IMAGES_RATIO
    dog_images_count = sample_images_count // SAMPLE_IMAGES_RATIO

    logger.info(
        "Found [ %s ] cat, [ %s ] dog, and [ %s ] sample images",
        cat_images_count,
        dog_images_count,
        sample_images_count,
    )

    return (
        cat_images[:cat_images_count],
        dog_images[:dog_images_count],
        sample_images[:sample_images_count],
    )


def write_to(source: list[Path], path: Path, logger: logging.Logger):
    path.mkdir(parents=True, exist_ok=True)

    img_count = len(source)
    dig_count = len(str(img_count))

    for i in range(img_count):
        it = source[i]

        if CATS_DATASET_CONTENT_PATH in str(it):
            img_type = "CAT"
        elif DOGS_DATASET_CONTENT_PATH in str(it):
            img_type = "DOG"
        else:
            img_type = "OTHER"

        new_file_name = path / f"{str(i).zfill(dig_count)}-{img_type}.jpg"
        img = Image.open(it)
        resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        resized.convert("RGB").save(new_file_name)
        if i % LOG_PERIOD == 0:
            logger.info("Prepared [ %s ] of [ %s ] images", i, img_count)


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    process_raw_dataset(logging.getLogger(__name__))
