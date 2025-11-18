import jax
import jax.numpy as jnp
import klax

class MSE(klax.Loss):
    """Mean Squared Error loss for batched model training."""

    def __call__(self, model, batch):
        x, y = batch
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y - y_pred))
