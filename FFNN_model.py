from typing import Callable

import equinox as eqx
import jax
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax
class HyperElasticFFNN(eqx.Module):
    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, key: PRNGKeyArray):
        self.layers = (
            klax.nn.Linear(6, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=key),
            klax.nn.Linear(16, 9, weight_init=he_normal(), key=key),  # final output
        )
        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x,
        )

    def __call__(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x
