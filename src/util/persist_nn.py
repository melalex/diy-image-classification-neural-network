from pathlib import Path
import pickle

from lib.neural_network import HyperParams, NeuralNetwork


def read_model(dir: Path, base_name: str, hyper_params: HyperParams) -> NeuralNetwork:
    path = create_model_path(dir, base_name, hyper_params)

    with path.open("rb") as source:
        return pickle.load(source)


def write_model(dir: Path, base_name: str, model: NeuralNetwork):
    dir.parent.mkdir(parents=True, exist_ok=True)

    path = create_model_path(dir, base_name, model.hyper_params)

    with path.open("wb") as dest:
        pickle.dump(model, dest)


def create_model_path(dir: Path, base_name: str, hyper_params: HyperParams):
    return (
        dir
        / f"{base_name}-{hyper_params.learning_rate}-{hyper_params.stop_condition.get_label}.pickle"
    )
