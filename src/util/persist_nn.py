from pathlib import Path
import pickle

from lib.neural_network import NeuralNetwork


def read_model(path: Path) -> NeuralNetwork:
    with path.open("rb") as source:
        return pickle.load(source)


def write_model(model: NeuralNetwork, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as dest:
        pickle.dump(model, dest)
