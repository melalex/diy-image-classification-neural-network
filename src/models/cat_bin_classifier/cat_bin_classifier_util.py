from pathlib import Path

import numpy as np

from src.definitions import IMAGE_HEIGHT, IMAGE_WIDTH
from src.util.image import read_all_images_and_predictions


def read_cat_bin_classifier_data_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x, y = read_all_images_and_predictions(path, IMAGE_WIDTH, IMAGE_HEIGHT)
    x = x[:,y != 2]
    y = y[y != 2]

    return x, y
