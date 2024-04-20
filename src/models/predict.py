import argparse
from pathlib import Path

import numpy as np

from definitions import (
    ANIMAL_CLASSIFIER_BASE_FILE_NAME,
    ANIMAL_CLASSIFIER_NAME,
    CAT_BIN_CLASSIFIER_NAME,
    IMAGE_VECTOR_SHAPE,
    MODELS_FOLDER,
)
from lib.neural_network import NeuralNetwork
from lib.persist_neural_network import read_model
from src.models.animal_classifier.animal_classifier_def import ANIMAL_HYPER_PARAMS
from src.models.cat_bin_classifier.cat_bin_classifier_def import CAT_BIN_HYPER_PARAMS
from src.util.image import prepare_image


def predict_with_model(path: Path, model: NeuralNetwork) -> np.ndarray:
    img = prepare_image(path).reshape((1, IMAGE_VECTOR_SHAPE)).T

    return model.predict(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="diy-image-classification-neural-network",
    )

    parser.add_argument("model")
    parser.add_argument("file")

    args = parser.parse_args()

    model = args.model
    file = Path(args.file)

    if model == ANIMAL_CLASSIFIER_NAME:
        model = read_model(
            MODELS_FOLDER, ANIMAL_CLASSIFIER_BASE_FILE_NAME, ANIMAL_HYPER_PARAMS
        )

        result = predict_with_model(file, model)

        print("Other = " + result[0][0])
        print("Cat = " + result[0][1])
        print("Dog = " + result[0][2])
    elif model == CAT_BIN_CLASSIFIER_NAME:
        model = read_model(MODELS_FOLDER, CAT_BIN_CLASSIFIER_NAME, CAT_BIN_HYPER_PARAMS)
        result = predict_with_model(file, model)

        if result > 0.5:
            print("It's a cat!!!")
        else:
            print("It's not a cat 😔")
    else:
        raise ValueError("Unknown model name " + model)
