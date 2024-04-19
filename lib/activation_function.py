import numpy as np


class ActivationFunction[T]:

    def apply(self, z: T) -> T:
        pass

    def applyDerivative(self, z: T) -> T:
        pass


class ReLuActivationFunction[T](ActivationFunction[T]):

    def apply(self, z: T) -> T:
        return (z > 0) * z

    def applyDerivative(self, z: T) -> T:
        return z > 0


class SigmoidActivationFunction[T](ActivationFunction[T]):

    def apply(self, z: T) -> T:
        return 1 / (1 + np.exp(-z))

    def applyDerivative(self, z: T) -> T:
        return self.apply(z) * (1 - self.apply(z))


class SoftMaxActivationFunction[T](ActivationFunction[T]):

    def apply(self, z: T) -> T:
        t = np.exp(z)
        return t / np.sum(t) 

    def applyDerivative(self, z: T) -> T:
        return 1


SIGMOID_ACTIVATION_FUN = SigmoidActivationFunction()
RELU_ACTIVATION_FUN = ReLuActivationFunction()
SOFT_MAX_ACTIVATION_FUN = SoftMaxActivationFunction()
