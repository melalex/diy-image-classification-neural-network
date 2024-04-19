from dataclasses import dataclass
from pathlib import Path
from PIL import Image

import numpy as np


def prepare_image(path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(path).resize((width, height))
    return normalize_image(np.array(img))


def image_to_vector(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).T


def normalize_image(x: np.ndarray) -> np.ndarray:
    return x / 255


def extract_prediction(path: Path) -> np.ndarray:
    stem = path.stem
    if "CAT" in stem:
        return np.array([0, 1, 0])
    elif "DOG" in stem:
        return np.array([0, 0, 1])
    else:
        return np.array([1, 0, 0])


def read_all_images_from(path: Path) -> list[Path]:
    imgs = [it for it in path.iterdir()]

    imgs.sort()

    return imgs


def read_all_images_and_predictions(path: Path, width: int, height: int):
    imgs = read_all_images_from(path)

    x = image_to_vector(np.stack([prepare_image(it, width, height) for it in imgs]))
    y = np.array([extract_prediction(it) for it in imgs]).reshape((3, -1))

    return x, y
