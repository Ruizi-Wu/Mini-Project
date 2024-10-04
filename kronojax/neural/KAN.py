# KAN: Kolmogorov-Arnold Networks
# https://arxiv.org/abs/2404.19756
 
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import normal


class ChebyKANLayer(nn.Module):
    """
    JAX port of:
    https://github.com/SynodicMonth/ChebyKAN

    Chebyshev KAN layer with output dimension `output_dim` 
    and Chebyshev polynomial degree `degree`.
    """
    output_dim: int
    degree: int

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        # initialize the coefficient so that the output
        # is of order one
        cheby_coeffs = self.param(
            'cheby_coeffs',
            normal(1 / jnp.sqrt(input_dim * (self.degree + 1))),
            (input_dim, self.output_dim, self.degree + 1)
        )
        arange = jnp.arange(0, self.degree + 1, 1)

        # Normalize x to [-1, 1] using tanh
        x = jnp.tanh(x)
        # View and repeat input degree + 1 times
        x = jnp.expand_dims(x, axis=-1)
        x = jnp.tile(x, (1, 1, self.degree + 1))  # shape = (batch_size, input_dim, degree + 1)
        # Compute Chebyshev polynomials
        # Recall that: T_n(cos(theta)) = cos(n * theta)
        x = jnp.cos(arange * jnp.arccos(x))
        # Compute the Chebyshev interpolation
        y = jnp.einsum('bid,iod->bo', x, cheby_coeffs)  # shape = (batch_size, output_dim)
        return y


class KAN(nn.Module):
    """
    Chebychev KAN network with dimension layer of dimension `dim_list` 
    and Chebyshev polynomial degree `degree`.

    Usage:
    =====
    # assume inputs of dimension D
    # and a regression task that consist in predicting
    # a scalar outpu, i.e. D_out = 1

    KAN_deg = 5             # degree of the Chebyshev polynomial
    D_out = 1               # output dimension
    dim_list = [100, 100, D_out]   # dimension of each layer (not including the input dim)
    kan = KAN(dim_list=dim_list, degree=KAN_deg)

    # initialize the network
    batch_sz = 32
    dummy_data = jnp.zeros((batch_sz, D))
    key, key_ = jr.split(key)
    params = kan.init(key_, dummy_data)
    """
    # dimension of each layer not including the input dimension
    dim_list: list
    degree: int

    @nn.compact
    def __call__(self, x):
        for dim_layer in self.dim_list[:-1]:
            x = ChebyKANLayer(dim_layer, self.degree)(x)
            x = nn.LayerNorm()(x)
        x = ChebyKANLayer(self.dim_list[-1], self.degree)(x)
        return x


class MLP(nn.Module):
    """
    Basic MLP with ReLU activation function.
    """
    # dimension of each layer not including the input dimension
    dim_list: list

    @nn.compact
    def __call__(self, x):
        for dim_layer in self.dim_list[:-1]:
            x = nn.Dense(dim_layer)(x)
            x = nn.elu(x)
        x = nn.Dense(self.dim_list[-1])(x)
        return x