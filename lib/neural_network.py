from dataclasses import dataclass
from logging import Logger
import logging
from os import path
from pathlib import Path
import pickle
from typing import Callable
import numpy as np

from lib.activation_function import ActivationFunction
from lib.cost_function import CostFunction
from lib.progress_tracker import ProgressTracker
from lib.stop_condition import StopCondition


@dataclass
class HyperParams:
    learning_rate: float


class NeuralNetworkLayer:

    def train(
        self, activations: np.ndarray, expected: np.ndarray, hyper_params: HyperParams
    ) -> tuple[np.ndarray, float]:
        pass

    def predict(self, activations: np.ndarray) -> np.ndarray:
        pass


class NeuralNetwork:
    __hyper_params: HyperParams
    __stop_condition: StopCondition
    __next: NeuralNetworkLayer
    __progress_tracker: ProgressTracker

    def __init__(
        self,
        hyper_params: HyperParams,
        stop_condition: StopCondition,
        next: NeuralNetworkLayer,
        progress_tracker: ProgressTracker,
    ) -> None:
        self.__hyper_params = hyper_params
        self.__stop_condition = stop_condition
        self.__next = next
        self.__progress_tracker = progress_tracker

    def train(
        self, activations: np.ndarray, expected: np.ndarray
    ) -> tuple[np.ndarray, float]:
        t = 0
        cost = 0
        da = 0

        while self.__stop_condition.test(t, cost):
            da, cost = self.__next.train(activations, expected, self.__hyper_params)
            self.__progress_tracker.track(t, cost)

        return da, cost

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return self.__next.predict(activations)


class HiddenLayer(NeuralNetworkLayer):
    __next: NeuralNetworkLayer
    __fun: ActivationFunction[np.ndarray]
    w: np.ndarray
    b: np.ndarray

    def __init__(
        self,
        next: NeuralNetworkLayer,
        fun: ActivationFunction[np.ndarray],
        w: np.ndarray,
        b: np.ndarray,
    ) -> None:
        self.__next = next
        self.__fun = fun
        self.w = w
        self.b = b

    def train(
        self, activations: np.ndarray, expected: np.ndarray, hyper_params: HyperParams
    ) -> tuple[np.ndarray, float]:
        z = np.dot(self.w, activations) + self.b
        a = self.__fun.apply(z)
        m = activations.shape[1]

        da_next, cost = self.__next.train(a, expected, hyper_params)

        dz = da_next * self.__fun.applyDerivative(z)
        dw = np.dot(dz, a.T) / m
        db = np.mean(dz, axis=1, keepdims=True)
        da = np.dot(self.w.T, dz)

        self.w = self.w - hyper_params.learning_rate * dw
        self.b = self.b - hyper_params.learning_rate * db

        return da, cost

    def predict(self, activations: np.ndarray) -> np.ndarray:
        z = np.dot(self.w, activations) + self.b
        a = self.fun.apply(z)
        return self.next.predict(a)


class OutputLayer(NeuralNetworkLayer):
    __cost_fun: CostFunction[np.ndarray]

    def __init__(
        self,
        cost_fun: CostFunction[np.ndarray],
    ) -> None:
        self.__cost_fun = cost_fun

    def train(
        self, activations: np.ndarray, expected: np.ndarray, _: HyperParams
    ) -> tuple[np.ndarray, float]:
        da = self.__cost_fun.applyDerivative(activations, expected)
        cost = self.__cost_fun.apply(activations, expected)

        return da, cost

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return activations
