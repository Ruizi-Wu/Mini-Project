import jax
import jax.numpy as jnp
from jax._src.typing import Array

#######################################
# TIME EMBEDDING WITH SINES
#######################################


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