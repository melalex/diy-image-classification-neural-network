import copy
from dataclasses import dataclass
import numpy as np

from lib.activation_function import ActivationFunction
from lib.cost_function import CostFunction
from lib.progress_tracker import NOOP_PROGRESS_TRACKER, ProgressTracker
from lib.stop_condition import (
    ApproximationStopCondition,
    IterCountStopCondition,
    StopCondition,
)


@dataclass
class HyperParams:
    learning_rate: float
    init_weight_scale: float = 0.01


class NeuralNetworkLayer:

    def train(
        self, activations: np.ndarray, expected: np.ndarray, hyper_params: HyperParams
    ) -> tuple[np.ndarray, float]:
        pass

    def predict(self, activations: np.ndarray) -> np.ndarray:
        pass


class HiddenLayer(NeuralNetworkLayer):
    w: np.ndarray
    b: np.ndarray
    dw: np.ndarray
    db: np.ndarray
    next: NeuralNetworkLayer
    fun: ActivationFunction[np.ndarray]

    def __init__(
        self,
        next: NeuralNetworkLayer,
        fun: ActivationFunction[np.ndarray],
        w: np.ndarray,
        b: np.ndarray,
    ) -> None:
        self.next = next
        self.fun = fun
        self.w = w
        self.b = b

    def __getstate__(self):
        return (self.next, self.fun, self.w, self.b)

    def __deepcopy__(self, memo):
        return self.__class__(
            copy.deepcopy(self.next, memo),
            self.fun,
            np.copy(self.w),
            np.copy(self.b),
        )

    def train(
        self, activations: np.ndarray, expected: np.ndarray, hyper_params: HyperParams
    ) -> tuple[np.ndarray, float]:
        z = np.dot(self.w, activations) + self.b
        a = self.fun.apply(z)
        m = activations.shape[1]

        da_next, cost = self.next.train(a, expected, hyper_params)

        dz = da_next * self.fun.applyDerivative(z)
        self.dw = np.dot(dz, activations.T) / m
        self.db = np.mean(dz, axis=1, keepdims=True)
        da = np.dot(self.w.T, dz)

        self.w = self.w - hyper_params.learning_rate * self.dw
        self.b = self.b - hyper_params.learning_rate * self.db

        return da, cost

    def predict(self, activations: np.ndarray) -> np.ndarray:
        z = np.dot(self.w, activations) + self.b
        a = self.fun.apply(z)
        return self.next.predict(a)


class OutputLayer(NeuralNetworkLayer):
    cost_fun: CostFunction[np.ndarray]

    def __init__(
        self,
        cost_fun: CostFunction[np.ndarray],
    ) -> None:
        self.cost_fun = cost_fun

    def train(
        self, activations: np.ndarray, expected: np.ndarray, _: HyperParams
    ) -> tuple[np.ndarray, float]:
        da = self.cost_fun.applyDerivative(activations, expected)
        cost = self.cost_fun.apply(activations, expected)

        return da, cost

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return activations


class NeuralNetwork:
    __first_layer: NeuralNetworkLayer
    __hyper_params: HyperParams
    __stop_condition: StopCondition
    __progress_tracker: ProgressTracker

    def builder():
        return NeuralNetworkBuilder()

    def __init__(
        self,
        hyper_params: HyperParams,
        stop_condition: StopCondition,
        first: NeuralNetworkLayer,
        progress_tracker: ProgressTracker,
    ) -> None:
        self.__first_layer = first
        self.__hyper_params = hyper_params
        self.__stop_condition = stop_condition
        self.__progress_tracker = progress_tracker

    def train(self, activations: np.ndarray, expected: np.ndarray) -> float:
        t = 0
        cost = 0

        while self.__stop_condition.test(t, cost):
            cost = self.__train_once(activations, expected)
            self.__progress_tracker.track(t, cost)
            t += 1

        return cost

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return self.__first_layer.predict(activations)

    def gradient_check(
        self,
        activations: np.ndarray,
        expected: np.ndarray,
        epsilon: float = 1e-7,
    ) -> float:
        self.__train_once(activations, expected)
        params_vector = self.__params_vector()
        grads_vector = self.__grads_vector()
        num_params = params_vector.shape[0]
        j_plus = np.zeros((num_params, 1))
        j_minus = np.zeros((num_params, 1))
        grad_approx = np.zeros((num_params, 1))
        builder = self.to_builder()

        for i in range(num_params):
            theta_plus = np.copy(params_vector)
            theta_plus[i] = theta_plus[i] + epsilon
            j_plus[i] = builder.build_from_params(theta_plus).__train_once(
                activations, expected
            )

            theta_minus = np.copy(params_vector)
            theta_minus[i] = theta_minus[i] - epsilon
            j_minus[i] = builder.build_from_params(theta_minus).__train_once(
                activations, expected
            )

            grad_approx[i] = (j_plus[i] - j_minus[i]) / (2 * epsilon)

            self.__progress_tracker.track_gradient_check(i, num_params)

        numerator = np.linalg.norm(grads_vector - grad_approx)
        denominator = np.linalg.norm(grads_vector) + np.linalg.norm(grad_approx)

        return numerator / denominator

    def to_builder(self):
        builder = (
            NeuralNetworkBuilder()
            .with_hyper_params(self.__hyper_params)
            .with_stop_condition(self.__stop_condition)
            .with_progress_tracker(self.__progress_tracker)
        )
        feature_count_not_set_flag = True
        runner = self.__first_layer

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
        runner = self.__first_layer

        while type(runner) is HiddenLayer:
            acc.append(runner)
            runner = runner.next

        return acc

    def __train_once(self, activations: np.ndarray, expected: np.ndarray) -> float:
        _, cost = self.__first_layer.train(activations, expected, self.__hyper_params)
        return cost

    def __params_vector(self) -> np.ndarray:
        acc = []
        runner = self.__first_layer

        while type(runner) is HiddenLayer:
            acc.append(runner.w.reshape(-1, 1))
            acc.append(runner.b.reshape(-1, 1))
            runner = runner.next

        return np.concatenate(acc, axis=None)

    def __grads_vector(self) -> np.ndarray:
        acc = []
        runner = self.__first_layer

        while type(runner) is HiddenLayer:
            assert hasattr(runner, "dw"), "gradients should be calculated first"
            assert hasattr(runner, "db"), "gradients should be calculated first"
            acc.append(runner.dw.reshape(-1, 1))
            acc.append(runner.db.reshape(-1, 1))
            runner = runner.next

        return np.concatenate(acc, axis=None)


class NeuralNetworkBuilder:

    @dataclass
    class __LayerDefinition:
        units_count: int
        fun: ActivationFunction[np.ndarray]

    __hyper_params: HyperParams
    __features_count: int
    __cost_fun: CostFunction[np.ndarray]
    __stop_condition: StopCondition
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

    def with_iter_count(self, iter_count: int):
        self.__stop_condition = IterCountStopCondition(iter_count)
        return self

    def with_accuracy(self, accuracy: float):
        self.__stop_condition = ApproximationStopCondition(accuracy)
        return self

    def with_stop_condition(self, stop_condition: StopCondition):
        self.__stop_condition = stop_condition
        return self

    def with_progress_tracker(self, progress_tracker: ProgressTracker):
        self.__progress_tracker = progress_tracker
        return self

    def with_layer(self, units_count, fun: ActivationFunction[np.ndarray]):
        self.__layers.append(NeuralNetworkBuilder.__LayerDefinition(units_count, fun))
        return self

    def build(self) -> NeuralNetwork:
        self.__assert_valid_state()

        next_layer = OutputLayer(self.__cost_fun)

        for i in range(len(self.__layers) - 1, -1, -1):
            curr = self.__layers[i]

            n_prev, n_curr = self.__extract_units_count(i)

            w = np.random.randn(n_curr, n_prev) * self.__hyper_params.init_weight_scale
            b = np.zeros((n_curr, 1))

            next_layer = HiddenLayer(next_layer, curr.fun, w, b)

        return NeuralNetwork(
            self.__hyper_params,
            self.__stop_condition,
            next_layer,
            self.__progress_tracker,
        )

    def build_from_params(self, params_vector: np.ndarray) -> NeuralNetwork:
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

        return NeuralNetwork(
            self.__hyper_params,
            self.__stop_condition,
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
        assert self.__stop_condition is not None, "stop_condition should be set"
        assert len(self.__layers) != 0, "at least 1 hidden layer should be provided"
