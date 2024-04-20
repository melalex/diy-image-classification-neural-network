import numpy as np


class CostFunction[T]:

    def apply(self, y_pred: T, y_true: T) -> float:
        pass

    def applyDerivative(self, y_pred: T, y_true: T) -> T:
        pass


class BinaryCrossEntropy[T](CostFunction[T]):

    def apply(self, y_pred: T, y_true: T) -> float:
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def applyDerivative(self, y_pred: T, y_true: T) -> T:
        return -(np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred))


class MultiClassCrossEntropy[T](CostFunction[T]):

    def apply(self, y_pred: T, y_true: T) -> float:
        return -np.sum(y_true * np.log(y_pred)).item() / y_true.shape[1]

    def applyDerivative(self, y_pred: T, y_true: T) -> T:
        return y_pred - y_true


BINARY_CLASSIFICATION_COST_FUN = BinaryCrossEntropy()
MULTI_CLASS_CLASSIFICATION_COST_FUN = MultiClassCrossEntropy()
