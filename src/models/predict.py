import argparse
from pathlib import Path

import numpy as np

from definitions import IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_PATH
from lib.neural_network import NeuralNetwork
from src.util.image import prepare_image
from src.util.persist_nn import read_model


def predict(path: Path) -> np.ndarray:
    return predict_with_model(path, read_model(MODEL_PATH))


def predict_with_model(path: Path, model: NeuralNetwork) -> np.ndarray:
    img = prepare_image(path).reshape((1, IMAGE_WIDTH * IMAGE_HEIGHT * 3)).T

    return model.predict(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="cats-binary-classification",
        description="Classifies whether supplied image contains cat or not",
    )

    parser.add_argument("filename")

    args = parser.parse_args()

    result = predict(Path(args.filename))

    print("Other = " + result[0][0])
    print("Cat = " + result[0][1])
    print("Dog = " + result[0][2])
