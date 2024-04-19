import numpy as np


class CostFunction[T]:

    def apply(self, actual: T, expected: T) -> float:
        pass

    def applyDerivative(self, actual: T, expected: T) -> T:
        pass


class BinaryCrossEntropy[T](CostFunction[T]):

    def apply(self, actual: T, expected: T) -> float:
        return -np.mean(expected * np.log(actual) + (1 - expected) * np.log(1 - actual))

    def applyDerivative(self, actual: T, expected: T) -> T:
        return -(np.divide(expected, actual) - np.divide(1 - expected, 1 - actual))


class MultiClassCrossEntropy[T](CostFunction[T]):

    def apply(self, actual: T, expected: T) -> float:
        loss = np.sum(expected * np.log(actual), axis=0)
        return -np.mean(loss)

    def applyDerivative(self, actual: T, expected: T) -> T:
        return actual - expected


BINARY_CLASSIFICATION_COST_FUN = BinaryCrossEntropy()
MULTI_CLASS_CLASSIFICATION_COST_FUN = MultiClassCrossEntropy()
