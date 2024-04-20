from dataclasses import dataclass
from logging import Logger
import logging
from pathlib import Path
import pickle
import numpy as np
from lib.activation_function import ActivationFunction
from lib.cost_function import CostFunction
from lib.neural_network import HiddenLayer, HyperParams, NeuralNetwork, OutputLayer
from lib.progress_tracker import NOOP_PROGRESS_TRACKER, ProgressTracker


class NeuralNetworkExt:
    hyper_params: HyperParams
    __delegate: NeuralNetwork
    __progress_tracker: ProgressTracker

    def builder():
        return NeuralNetworkExtBuilder()

    def __init__(
        self,
        hyper_params: HyperParams,
        first: NeuralNetwork,
        progress_tracker: ProgressTracker,
    ) -> None:
        self.__delegate = first
        self.hyper_params = hyper_params
        self.__progress_tracker = progress_tracker

    def train(self, x: np.ndarray, y_true: np.ndarray) -> float:
        t = 0
        cost = 0

        while self.hyper_params.stop_condition.test(t, cost):
            cost = self.__train_once(x, y_true)
            self.__progress_tracker.track(t, cost)
            t += 1

        return cost

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.__delegate.predict(x)

    def gradient_check(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        epsilon: float = 1e-7,
    ) -> float:
        self.__train_once(x, y_true)
        params_vector = self.__params_vector()
        grads_vector = self.__grads_vector()
        num_params = params_vector.shape[0]
        grad_approx = np.zeros((num_params, 1))
        builder = self.to_builder()

        for i in range(num_params):
            old_value = params_vector[i]
            params_vector[i] = old_value + epsilon
            nn_plus = builder.build_from_params(params_vector)
            j_plus = nn_plus.__forward_propagation(x, y_true)

            params_vector[i] = old_value - epsilon
            nn_minus = builder.build_from_params(params_vector)
            j_minus = nn_minus.__forward_propagation(x, y_true)
            params_vector[i] = old_value

            grad_approx[i] = (j_plus - j_minus) / (2 * epsilon)

            self.__progress_tracker.track_gradient_check(i, num_params)

        numerator = np.linalg.norm(grads_vector - grad_approx)
        denominator = np.linalg.norm(grads_vector) + np.linalg.norm(grad_approx)

        return numerator / denominator

    def to_builder(self):
        builder = (
            NeuralNetworkExtBuilder()
            .with_hyper_params(self.hyper_params)
            .with_progress_tracker(self.__progress_tracker)
        )
        feature_count_not_set_flag = True
        runner = self.__delegate

        while runner is not None:
            if type(runner) is HiddenLayer:
                if feature_count_not_set_flag:
                    builder.with_feature_count(runner.w.shape[1])
                    feature_count_not_set_flag = False

                builder.with_layer(runner.w.shape[0], runner.fun)
                runner = runner.next
            elif type(runner) is OutputLayer:
                builder.with_cost_fun(runner.cost_fun)
                runner = None

        return builder

    def hidden_layers(self) -> list[HiddenLayer]:
        acc = []
        runner = self.__delegate

        while type(runner) is HiddenLayer:
            acc.append(runner)
            runner = runner.next

        return acc

    def __forward_propagation(self, x: np.ndarray, y_true: np.ndarray) -> float:
        _, cost = self.__delegate.forward_propagation(x, y_true, self.hyper_params)
        return cost

    def __train_once(self, x: np.ndarray, y_true: np.ndarray) -> float:
        _, cost = self.__delegate.train(x, y_true, self.hyper_params)
        return cost

    def __params_vector(self) -> np.ndarray:
        acc = []
        runner = self.__delegate

        while type(runner) is HiddenLayer:
            acc.append(runner.w.reshape((-1, 1)))
            acc.append(runner.b.reshape((-1, 1)))
            runner = runner.next

        return np.concatenate(acc)

    def __grads_vector(self) -> np.ndarray:
        acc = []
        runner = self.__delegate

        while type(runner) is HiddenLayer:
            assert hasattr(runner, "dw"), "gradients should be calculated first"
            assert hasattr(runner, "db"), "gradients should be calculated first"
            acc.append(runner.dw.reshape((-1, 1)))
            acc.append(runner.db.reshape((-1, 1)))
            runner = runner.next

        return np.concatenate(acc)


class NeuralNetworkExtBuilder:

    @dataclass
    class __LayerDefinition:
        units_count: int
        fun: ActivationFunction[np.ndarray]

    __hyper_params: HyperParams
    __features_count: int
    __cost_fun: CostFunction[np.ndarray]
    __progress_tracker: ProgressTracker
    __layers: list[__LayerDefinition]

    def __init__(self) -> None:
        self.__features_count = 0
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

    def with_progress_tracker(self, progress_tracker: ProgressTracker):
        self.__progress_tracker = progress_tracker
        return self

    def with_layer(self, units_count, fun: ActivationFunction[np.ndarray]):
        self.__layers.append(
            NeuralNetworkExtBuilder.__LayerDefinition(units_count, fun)
        )
        return self

    def build(self) -> NeuralNetworkExt:
        self.__assert_valid_state()

        next_layer = OutputLayer(self.__cost_fun)

        for i in range(len(self.__layers) - 1, -1, -1):
            curr = self.__layers[i]

            n_prev, n_curr = self.__extract_units_count(i)

            w = np.random.randn(n_curr, n_prev) * self.__hyper_params.init_weight_scale
            b = np.zeros((n_curr, 1))

            next_layer = HiddenLayer(next_layer, curr.fun, w, b)

        return NeuralNetworkExt(
            self.__hyper_params,
            next_layer,
            self.__progress_tracker,
        )

    def build_from_params(self, params_vector: np.ndarray) -> NeuralNetworkExt:
        self.__assert_valid_state()

        next_layer = OutputLayer(self.__cost_fun)
        params_vector_pos = params_vector.shape[0]

        for i in range(len(self.__layers) - 1, -1, -1):
            curr = self.__layers[i]

            n_prev, n_curr = self.__extract_units_count(i)
            b_pos = params_vector_pos - n_curr
            w_pos = b_pos - n_curr * n_prev
            b = params_vector[b_pos:params_vector_pos].reshape((n_curr, 1))
            w = params_vector[w_pos:b_pos].reshape((n_curr, n_prev))
            params_vector_pos = w_pos

            next_layer = HiddenLayer(next_layer, curr.fun, w, b)

        return NeuralNetworkExt(
            self.__hyper_params,
            next_layer,
            self.__progress_tracker,
        )

    def __extract_units_count(self, layer: int):
        curr = self.__layers[layer]

        if layer == 0:
            prev = self.__features_count
        else:
            prev = self.__layers[layer - 1].units_count

        return prev, curr.units_count

    def __assert_valid_state(self):
        assert self.__hyper_params is not None, "hyper_params should be set"
        assert self.__features_count != 0, "features_count should be set"
        assert self.__cost_fun is not None, "cost_fun should be set"
        assert len(self.__layers) != 0, "at least 1 hidden layer should be provided"
