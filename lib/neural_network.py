import copy
from dataclasses import dataclass
import numpy as np

from lib.activation_function import ActivationFunction
from lib.cost_function import CostFunction
from lib.stop_condition import StopCondition


@dataclass
class HyperParams:
    learning_rate: float
    stop_condition: StopCondition
    beta: float = 0
    batch_size: int = -1
    init_weight_scale: float = 0.01


class NeuralNetwork:

    def train(
        self, a_prev: np.ndarray, y_true: np.ndarray, hyper_params: HyperParams
    ) -> tuple[np.ndarray, float]:
        pass

    def forward_propagation(
        self, a_prev: np.ndarray, y_true: np.ndarray, hyper_params: HyperParams
    ) -> tuple[np.ndarray, float]:
        pass

    def predict(self, a_prev: np.ndarray) -> np.ndarray:
        pass


class HiddenLayer(NeuralNetwork):
    w: np.ndarray
    b: np.ndarray
    dw: np.ndarray
    db: np.ndarray
    dw_velocity: np.ndarray
    db_velocity: np.ndarray
    next: NeuralNetwork
    fun: ActivationFunction[np.ndarray]

    def __init__(
        self,
        next: NeuralNetwork,
        fun: ActivationFunction[np.ndarray],
        w: np.ndarray,
        b: np.ndarray,
    ) -> None:
        self.next = next
        self.fun = fun
        self.w = w
        self.b = b
        self.dw_velocity = np.zeros(w.shape)
        self.db_velocity = np.zeros(b.shape)

    def __getstate__(self):
        return (self.next, self.fun, self.w, self.b)

    def __setstate__(self, state):
        self.next, self.fun, self.w, self.b = state

    def __deepcopy__(self, memo):
        return self.__class__(
            copy.deepcopy(self.next, memo),
            self.fun,
            np.copy(self.w),
            np.copy(self.b),
        )

    def train(
        self, a_prev: np.ndarray, y_true: np.ndarray, hyper_params: HyperParams
    ) -> tuple[np.ndarray, float]:
        z = np.dot(self.w, a_prev) + self.b
        a = self.fun.apply(z)
        m = a_prev.shape[1]

        da_next, cost = self.next.train(a, y_true, hyper_params)

        dz = da_next * self.fun.applyDerivative(z)
        self.dw = np.dot(dz, a_prev.T) / m
        self.db = np.mean(dz, axis=1, keepdims=True)
        da = np.dot(self.w.T, dz)

        self.dw_velocity = (
            hyper_params.beta * self.dw_velocity + (1 - hyper_params.beta) * self.dw
        )
        self.db_velocity = (
            hyper_params.beta * self.db_velocity + (1 - hyper_params.beta) * self.db
        )
        self.w -= hyper_params.learning_rate * self.dw_velocity
        self.b -= hyper_params.learning_rate * self.db_velocity

        return da, cost

    def forward_propagation(
        self, a_prev: np.ndarray, y_true: np.ndarray, hyper_params: HyperParams
    ) -> tuple[np.ndarray, float]:
        z = np.dot(self.w, a_prev) + self.b
        a = self.fun.apply(z)

        return self.next.forward_propagation(a, y_true, hyper_params)

    def predict(self, a_prev: np.ndarray) -> np.ndarray:
        z = np.dot(self.w, a_prev) + self.b
        a = self.fun.apply(z)
        return self.next.predict(a)


class OutputLayer(NeuralNetwork):
    cost_fun: CostFunction[np.ndarray]

    def __init__(
        self,
        cost_fun: CostFunction[np.ndarray],
    ) -> None:
        self.cost_fun = cost_fun

    def train(
        self, a_prev: np.ndarray, y_true: np.ndarray, hyper_params: HyperParams
    ) -> tuple[np.ndarray, float]:
        return self.forward_propagation(a_prev, y_true, hyper_params)

    def forward_propagation(
        self, a_prev: np.ndarray, y_true: np.ndarray, _: HyperParams
    ) -> tuple[np.ndarray, float]:
        da = self.cost_fun.applyDerivative(a_prev, y_true)
        cost = self.cost_fun.apply(a_prev, y_true)

        return da, cost

    def predict(self, a_prev: np.ndarray) -> np.ndarray:
        return a_prev
