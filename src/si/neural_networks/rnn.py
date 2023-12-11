from copy import deepcopy
from typing import Tuple

import numpy as np

from si.neural_networks.activation import TanhActivation, ActivationLayer
from si.neural_networks.layers import Layer
from si.neural_networks.optimizers import SGD


class RNN(Layer):
    """A Vanilla Fully-Connected Recurrent Neural Network layer."""

    def __init__(self, n_units: int, activation: ActivationLayer = None, bptt_trunc: int = 5,
                 input_shape: Tuple = None):
        """
        Initializes the layer.

        Parameters
        ----------
        n_units: int
            The number of units in the layer (i.e. the number of hidden states).
        activation: ActivationLayer
            The activation function to apply to the output of each state.
        bptt_trunc: int
            The number of time steps to backpropagate through time (i.e. the number of time steps to unroll the RNN).
        input_shape: Tuple
            The shape of the input to the layer.
        """
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = TanhActivation() if activation is None else activation
        self.bptt_trunc = bptt_trunc

        self.W = None  # Weight of the previous state
        self.V = None  # Weight of the output
        self.U = None  # Weight of the input

    def initialize(self, optimizer):
        """
        Initializes the weights of the layer.

        Parameters
        ----------
        optimizer: Optimizer
            The optimizer to use for updating the weights.
        """
        timesteps, input_dim = self.input_shape
        # Initialize the weights
        limit = 1 / np.sqrt(input_dim)
        self.U = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / np.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (input_dim, self.n_units))
        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        # Weight optimizers
        self.U_opt = deepcopy(optimizer)
        self.V_opt = deepcopy(optimizer)
        self.W_opt = deepcopy(optimizer)

    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.layer_input = input
        batch_size, timesteps, input_dim = input.shape

        # Save these values for use in backprop.
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        # Set last time step to zero for calculation of the state_input at time step zero (already zero?)
        # self.states[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):
            # Input to state_t is the current input and output of previous states
            self.state_input[:, t] = input[:, t].dot(self.U.T) + self.states[:, t - 1].dot(self.W.T)
            self.states[:, t] = self.activation.activation_function(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward_propagation(self, accum_grad: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        accum_grad: numpy.ndarray
            The accumulated gradient from the previous layer.
        Returns:
        --------
        numpy.ndarray
            The accumulated gradient w.r.t the input of the layer.
        """
        _, timesteps, _ = accum_grad.shape

        # Variables where we save the accumulated gradient w.r.t each parameter
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        # The gradient w.r.t the layer input.
        # Will be passed on to the previous layer in the network
        accum_grad_next = np.zeros_like(accum_grad)

        # Back Propagation Through Time
        for t in reversed(range(timesteps)):
            # Update gradient w.r.t V at time step t
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            # Calculate the gradient w.r.t the state input
            grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.derivative(self.state_input[:, t])
            # Gradient w.r.t the layer input
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            # Update gradient w.r.t W and U by backprop. from time step t for at most
            # self.bptt_trunc number of time steps
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t + 1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                # Calculate gradient w.r.t previous state
                grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.derivative(self.state_input[:, t_ - 1])

        # Update weights
        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)

        return accum_grad_next

    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return self.input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)


if __name__ == '__main__':
    data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    rnn = RNN(10, input_shape=(3, 3))
    rnn.initialize(SGD())
    print(rnn.forward_propagation(data))
    print(rnn.backward_propagation(data))
