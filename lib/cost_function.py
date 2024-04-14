import numpy as np


class CostFunction[T]:

    def apply(self, actual: T, expected: T) -> float:
        pass

    def applyDerivative(self, actual: T, expected: T) -> T:
        pass


class BinaryClassificationCostFunction[T](CostFunction[T]):

    def apply(self, actual: T, expected: T) -> float:
        return -np.mean(expected * np.log(actual) + (1 - expected) * np.log(1 - actual))

    def applyDerivative(self, actual: T, expected: T) -> T:
        return -(np.divide(expected, actual) - np.divide(1 - expected, 1 - actual))

BINARY_CLASSIFICATION_COST_FUN = BinaryClassificationCostFunction()
