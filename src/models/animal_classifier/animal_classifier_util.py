from pathlib import Path

import numpy as np

from src.definitions import IMAGE_HEIGHT, IMAGE_WIDTH
from src.util.image import read_all_images_and_predictions


def read_animal_classifier_data_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x, y = read_all_images_and_predictions(path, IMAGE_WIDTH, IMAGE_HEIGHT)
    y_one_hot = np.zeros((3, y.size))
    y_one_hot[y, np.arange(y.size)] = 1

    return x, y_one_hot
