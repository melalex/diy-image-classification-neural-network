from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from PIL import Image

import numpy as np

from src.definitions import CATS_DATASET_CONTENT_PATH, DOGS_DATASET_CONTENT_PATH


def prepare_image(path: Path, width: int, height: int):
    img = Image.open(path).resize((width, height))
    return normalize_image(np.array(img))


def image_to_vector(x: np.ndarray):
    return x.reshape(x.shape[0], -1).T


def normalize_image(x: np.ndarray):
    return x / 255


def extract_label(path: Path):
    stem = path.stem
    if "CAT" in stem:
        return 1
    elif "DOG" in stem:
        return 2
    else:
        return 0


def create_labeled_file_name(i: int, dig_count: int, source: Path, dir: Path):
    if CATS_DATASET_CONTENT_PATH in str(source):
        img_type = "CAT"
    elif DOGS_DATASET_CONTENT_PATH in str(source):
        img_type = "DOG"
    else:
        img_type = "OTHER"

    return dir / f"{str(i).zfill(dig_count)}-{img_type}.jpg"


def read_all_images_from(path: Path):
    imgs = [it for it in path.iterdir()]

    imgs.sort()

    return imgs


def read_all_images_and_predictions(path: Path, width: int, height: int):
    imgs = read_all_images_from(path)

    x = image_to_vector(np.stack([prepare_image(it, width, height) for it in imgs]))
    y = np.array([extract_label(it) for it in imgs])

    return x, y
