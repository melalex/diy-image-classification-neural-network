from dataclasses import dataclass
from typing import Callable
import numpy as np

from lib.activation_function import ActivationFunction
from lib.cost_function import CostFunction
from lib.neural_network import HiddenLayer, HyperParams, NeuralNetwork, OutputLayer
from lib.progress_tracker import NOOP_PROGRESS_TRACKER, ProgressTracker
from lib.stop_condition import (
    ApproximationStopCondition,
    IterCountStopCondition,
    StopCondition,
)


@dataclass
class LayerDefinition:
    units_count: int
    fun: ActivationFunction[np.ndarray]


class NeuralNetworkBuilder:
    __hyper_params: HyperParams
    __features_count: int
    __cost_fun: CostFunction[np.ndarray]
    __stop_condition: StopCondition
    __progress_tracker: ProgressTracker
    __layers: list[LayerDefinition]

    def __init__(self) -> None:
        self.__progress_tracker = NOOP_PROGRESS_TRACKER
        self.__layers = []

    def with_hyper_params(self, hyper_params: HyperParams):
        self.__hyper_params = hyper_params
        return self

    def with_feature_count(self, features_count: int):
        self.__features_count = features_count
        return self

    def with_cost_fun(self, cost_fun: CostFunction[np.ndarray]):
        self.__cost_fun = cost_fun
        return self

    def with_iter_count(self, iter_count: int):
        self.__stop_condition = IterCountStopCondition(iter_count)
        return self

    def with_accuracy(self, accuracy: float):
        self.__stop_condition = ApproximationStopCondition(accuracy)
        return self

    def with_progress_tracker(self, progress_tracker: ProgressTracker):
        self.__progress_tracker = progress_tracker
        return self

    def with_layer(self, units_count, fun: ActivationFunction[np.ndarray]):
        self.__layers.append(LayerDefinition(units_count, fun))
        return self

    def build(self):
        assert self.__hyper_params is None, "hyper_params should be set"
        assert self.__features_count is None, "features_count should be set"
        assert self.__cost_fun is None, "cost_fun should be set"
        assert self.__stop_condition is None, "stop_condition should be set"

        next_layer = OutputLayer(self.__cost_fun)

        for i in range(len(self.__layers), 0, -1):
            curr = self.__layers[i]

            if i == 0:
                prev = self.__features_count
            else:
                prev = self.__layers[i - 1].units_count

            w, b = self.__init_params(curr.units_count, prev)

            next_layer = HiddenLayer(next_layer, curr.fun, w, b)

        return NeuralNetwork(
            self.__hyper_params,
            self.__stop_condition,
            next_layer,
            self.__progress_tracker,
        )

    def __init_params(curr: int, prev: int):
        w1 = np.random.randn(curr, prev) * 0.01
        b1 = np.zeros((curr, 1))

        return w1, b1
