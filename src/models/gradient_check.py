import logging
import logging.config
from pathlib import Path
from src.definitions import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEARNING_RATE,
    LOGGING_CONFIG_PATH,
    TEST_DATA_FOLDER,
)
from src.models.neural_network_def import create_neural_network
from src.util.image import read_all_images_and_predictions


def gradient_check(dataset: Path, learning_factor: float, logger: logging.Logger) -> float:
    x, y = read_all_images_and_predictions(dataset, IMAGE_WIDTH, IMAGE_HEIGHT)

    logger.info("Loaded [ %s ] images", x.shape[1])

    nn = create_neural_network(x.shape[0], learning_factor, 0)

    gradient_check_diff = nn.gradient_check(x, y)

    logger.info("Gradient check result: " + str(gradient_check_diff))

    return gradient_check_diff


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    gradient_check(
        TEST_DATA_FOLDER,
        LEARNING_RATE,
        logging.getLogger(__name__),
    )
