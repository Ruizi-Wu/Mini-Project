import jax
import optax
import time

import jax.numpy as jnp
import jax.random as jrd
import numpy as np

from jax.random import PRNGKey
from flax import linen as nn
import jax.numpy as jnp
from flax.linen.initializers import normal
from jax._src.typing import Array

import sys
sys.path.insert(1, '..')



def time_embedding_single(t: float,
                          freq_min: float,
                          freq_max: float,
                          embedding_dim: int):
    """ time embedding """
    freqs = jnp.exp(jnp.linspace(jnp.log(freq_min),
                                 jnp.log(freq_max),
                                 embedding_dim//2))
    t_times_freqs = t*freqs
    t_sines = jnp.sin(t_times_freqs)
    t_cosines = jnp.cos(t_times_freqs)
    return jnp.concatenate([t_sines, t_cosines])


time_embedding_ = jax.vmap(time_embedding_single,
                           in_axes=(0, None, None, None))


def time_embedding(t_arr: Array,
                   freq_min: float,
                   freq_max: float,
                   embedding_dim: int):
    """ time embedding

    Args:
        t_arr (Array): array of times to embed with time features
        freq_min: smallest frequency
        freq_max: largest frequency
        embedding_dim: total amount of features after the embedding

    Returns:
        features: Array
    """
    return time_embedding_(t_arr,
                           freq_min,
                           freq_max,
                           embedding_dim)


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
        # return x
        return nn.softplus(x)


key = jrd.PRNGKey(0)


def log_target(x):
    """ Mixture of two Gaussians """
    mu1 = -1.
    mu2 = 1.
    sigma2 = 0.09
    proba_1 = jnp.exp(-0.5*(x-mu1)**2 / sigma2)
    proba_2 = jnp.exp(-0.5*(x-mu2)**2 / sigma2)
    return jnp.log(0.5 * proba_1 + 0.5 * proba_2)


class NN_drift(nn.Module):
    """parametrize drift function"""

    time_embedding_dim: int
    time_freq_min: float
    time_freq_max: float
    dim_list: list

    @nn.compact
    def __call__(self, x, t):

        t_embedded = time_embedding(t, self.time_freq_min, self.time_freq_max, self.time_embedding_dim)
        input_forward = jnp.concatenate([x, t_embedded], axis=1)
        NN_forward = KAN(dim_list=self.dim_list, degree=2)
        # NN_forward = MLP(dim_list=self.dim_list)

        return NN_forward(input_forward)

class NN_flow(nn.Module):
    """parametrize flow function"""

    time_embedding_dim: int
    time_freq_min: float
    time_freq_max: float
    dim_list: list

    @nn.compact
    def __call__(self, x, t):

        t_embedded = time_embedding(t, self.time_freq_min, self.time_freq_max, self.time_embedding_dim)
        input_flow = jnp.concatenate([x, t_embedded], axis=1)
        NN_flow = MLP(dim_list=self.dim_list)

        return NN_flow(input_flow)
    
class NN_FlowAndDirft(nn.Module):
    """concatenate two neural networks for drift and flow"""

    time_embedding_dim: int
    time_freq_min: float
    time_freq_max: float
    dim_list_drift: list
    dim_list_flow: list

    @nn.compact
    def __call__(self, x, t):

        drift = NN_drift(time_embedding_dim=self.time_embedding_dim, 
                         time_freq_min=self.time_freq_min, time_freq_max=self.time_freq_max, dim_list=self.dim_list_drift)
        
        flow = NN_flow(time_embedding_dim=self.time_embedding_dim, 
                       time_freq_min=self.time_freq_min, time_freq_max=self.time_freq_max, dim_list=self.dim_list_flow)

        return drift(x, t), flow(x, t)
    

key, subkey = jrd.split(key)
time_embedding_dim = 16
time_freq_min = 1.
time_freq_max = 10
output_dim = 1
dim_list_drift = [32, 32, output_dim]
dim_list_flow = [32, 32, output_dim]
GFnet = NN_FlowAndDirft(time_embedding_dim, time_freq_min, time_freq_max, dim_list_drift, dim_list_flow)

# initialize the network
batch_sz = 32
key, subkey = jrd.split(key)
xs = jrd.normal(subkey, (batch_sz, 1))
ts = jrd.uniform(subkey, (batch_sz, 1))
params = GFnet.init(subkey, xs, ts)
    

def log_normal_density(x, mu, sigma2):
    """
    x: n-dim array
    mu: n-dim array
    sigma2: scalar
    """
    return - 0.5 * jnp.log(2 * jnp.pi * sigma2) - 0.5 * (x - mu)**2 / sigma2


def Traj(
        params: any,
        batch_sz: int,
        N_step: int,
        key: PRNGKey
        ):
    """keep track on flow function, forward probability, and backward probability"""

    T = 1.
    dt = T / N_step
    sqt = jnp.sqrt(dt)

    def _step(carry, _):
        xo, t, count, key = carry
        coeff = count / (count + 1)
        to = jnp.ones_like(xo) * t
        u, f = GFnet.apply(params, xo, to)
        key, subkey = jrd.split(key)
        dw = sqt * jrd.normal(subkey, xo.shape)
        xn = xo + u*dt + dw
        t += dt
        count += 1
        log_PF = log_normal_density(xn, xo + u*dt, dt) # Forward probability
        log_PB = jnp.where(
            coeff == 0,
            log_normal_density(xo, coeff * xn, 0.5 * dt), # prevent zero variance
            log_normal_density(xo, coeff * xn, coeff * dt) # Backward probability
            )
        output_dict = {
            "t": t,
            "x": xo,
            "log(P_forward)": log_PF,
            "log(P_backward)": log_PB,
            "state flow": f
            }
        return (xn, t, count, subkey), output_dict
    
    key, subkey = jrd.split(key)
    t_init = 0.
    x_init = jnp.zeros((batch_sz, 1))
    count_init = 0
    carry_init = (x_init, t_init, count_init, subkey)
    _, trajectory = jax.lax.scan(_step, carry_init, xs = None, length=N_step)
    return trajectory


def loss(
        params, 
        batch_sz: int, 
        N_step: int, 
        key: PRNGKey
        ):
    """loss function -- total trajectory balance"""
    
    trajectory = Traj(params, batch_sz, N_step, key)
    F0 = trajectory["state flow"][0]
    xN = trajectory["state flow"][-1]
    log_uN = log_target(xN)
    ratio1 = jnp.sum(trajectory["log(P_forward)"] - trajectory["log(P_backward)"], axis=0)
    ratio2 = jnp.log(F0) - log_uN + ratio1
    loss = jnp.mean(ratio2**2, axis=0)
    return loss.reshape(())

loss = jax.jit(loss, static_argnums=(1, 2,)) 
loss_value_grad = jax.value_and_grad(loss, argnums=0)


lr = 1e-3
optimizer = optax.adam(learning_rate=lr)

opt_state = optimizer.init(params)

def update(params, opt_state, batch_sz, N_step, key):
    """update the parameters"""

    loss_value, grads = loss_value_grad(params, N_step, batch_sz, key)
    # loss_value = loss(params, batch_sz, N_step, key)
    # grads = jax.grad(loss)(params, batch_sz, N_step, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    # print(grads)
    return new_params, opt_state, loss_value, grads

update = jax.jit(update, static_argnums=(2, 3))


Niter = 10
Batch_SZ = 256
LR = 10**-2
optimizer = optax.adam(learning_rate=LR)

loss_values = []

time_start = time.time()
for i in range(Niter):
    key, subkey = jrd.split(key)
    params, opt_state, loss_value, grads = update(params, opt_state, Batch_SZ, 10, subkey)
    loss_values.append(loss_value)
    # print(f"Iteration {i}, Loss {loss_value}")
    # if i % 10 == 0:
    #     time_current = time.time()
    #     print(f"Iteration {i:4d}/{Niter}  |  Loss: {loss_value:.2f}  |  Time: {time_current - time_start:.2f} s")