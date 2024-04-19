import unittest

import numpy as np

from lib.activation_function import RELU_ACTIVATION_FUN, SIGMOID_ACTIVATION_FUN
from lib.cost_function import MULTI_CLASS_CLASSIFICATION_COST_FUN
from lib.neural_network import HyperParams, NeuralNetworkBuilder


class TestNeuralNetwork(unittest.TestCase):

    def test_build(self):
        iter_count = 100
        learning_rate = 0.05
        nn = (
            NeuralNetworkBuilder()
            .with_iter_count(iter_count)
            .with_hyper_params(HyperParams(learning_rate))
            .with_feature_count(5)
            .with_cost_fun(MULTI_CLASS_CLASSIFICATION_COST_FUN)
            .with_layer(3, RELU_ACTIVATION_FUN)
            .with_layer(1, SIGMOID_ACTIVATION_FUN)
            .build()
        )

        hidden_layers = nn.hidden_layers()

        w1 = hidden_layers[0].w
        b1 = hidden_layers[0].b
        w2 = hidden_layers[1].w
        b2 = hidden_layers[1].b

        self.assertEqual(w1.shape, (3, 5))
        self.assertEqual(b1.shape, (3, 1))
        self.assertEqual(w2.shape, (1, 3))
        self.assertEqual(b2.shape, (1, 1))

    def test_build_from_prototype(self):
        iter_count = 100
        learning_rate = 0.05
        proto_nn = (
            NeuralNetworkBuilder()
            .with_iter_count(iter_count)
            .with_hyper_params(HyperParams(learning_rate))
            .with_feature_count(5)
            .with_cost_fun(MULTI_CLASS_CLASSIFICATION_COST_FUN)
            .with_layer(3, RELU_ACTIVATION_FUN)
            .with_layer(1, SIGMOID_ACTIVATION_FUN)
            .build()
        )

        proto_hidden_layers = proto_nn.hidden_layers()

        proto_hidden_layers[0].w = np.array([it for it in range(15)]).reshape((3, 5))
        proto_hidden_layers[0].b = np.array([0, 1, 2]).reshape((3, 1))
        proto_hidden_layers[1].w = np.array([0, 1, 2]).reshape((1, 3))
        proto_hidden_layers[1].b = np.array([0]).reshape((1, 1))

        nn = proto_nn.to_builder().build_from_params_vector(proto_nn.params_vector())

        hidden_layers = nn.hidden_layers()

        self.assertTrue(np.array_equal(hidden_layers[0].w, proto_hidden_layers[0].w))
        self.assertTrue(np.array_equal(hidden_layers[0].b, proto_hidden_layers[0].b))
        self.assertTrue(np.array_equal(hidden_layers[1].w, proto_hidden_layers[1].w))
        self.assertTrue(np.array_equal(hidden_layers[1].b, proto_hidden_layers[1].b))


if __name__ == "__main__":
    unittest.main()
