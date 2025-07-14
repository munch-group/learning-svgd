import jax
import jax.numpy as jnp
import equinox as eqx



class LessThanOneDecoder(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        self.linear = eqx.nn.Linear(latent_dim, 3, key=key)

    def __call__(self, z: jnp.ndarray):
        probs = jax.nn.softmax(self.linear(z))
        return probs[0], probs[1]  # a, b ∈ (0,1), a + b < 1
    
class SumToOneDecoder(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        self.linear = eqx.nn.Linear(latent_dim, 1, key=key)

    def __call__(self, z: jnp.ndarray):
        s = jax.nn.sigmoid(self.linear(z)[0])
        return s, 1.0 - s

class IndependentProbDecoder(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        self.linear = eqx.nn.Linear(latent_dim, 2, key=key)

    def __call__(self, z: jnp.ndarray):
        logits = self.linear(z)  # shape (2,)
        a, b = jax.nn.sigmoid(logits[0]), jax.nn.sigmoid(logits[1])
        return a, b