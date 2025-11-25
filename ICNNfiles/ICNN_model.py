from typing import Callable

import equinox as eqx
import jax
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax
# ICNN modules
class PositiveLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    def __init__(self, in_features, out_features, key):
        w_key, b_key = jrandom.split(key)
        self.weight = jax.nn.initializers.he_normal()(w_key, (out_features, in_features))
        self.bias = jax.nn.initializers.zeros(b_key, (out_features,))
    def __call__(self, x):
        return jnp.abs(self.weight) @ x + self.bias

class PhysicsAugmentedICNN(eqx.Module):
    layers: List[PositiveLinear]
    def __init__(self, key):
        k1, k2, k3 = jrandom.split(key, 3)
        self.layers = [
            PositiveLinear(5, 16, key=k1),  # Input: 5 invariants
            PositiveLinear(16, 16, key=k2),
            PositiveLinear(16, 1, key=k3),  # Output: 1 energy
        ]
    def __call__(self, F):
        I1, J, I4, I5 = invariants(F)
        x = jnp.stack([I1, J, -J, I4, I5])
        for layer in self.layers[:-1]:
            x = jax.nn.softplus(layer(x))
        x = self.layers[-1](x)
        return x[0]

