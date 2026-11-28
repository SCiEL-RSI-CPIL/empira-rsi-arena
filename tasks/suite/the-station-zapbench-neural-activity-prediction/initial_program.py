"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Shared-neuron MLP: the Station's ZAPBench baseline (its system defaults, written out)."""
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

BATCH_SIZE = 8


def _define_hyperparameters():
    return {"learning_rate": 0.001}


class SharedNeuronMLP(nn.Module):
    """Every neuron's 4-step context goes through the same MLP (4 -> 64 -> 32) with BatchNorm and Dropout."""
    hidden_size: int = 64
    output_horizon: int = 32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        batch_size, input_horizon, num_neurons = x.shape
        x = jnp.transpose(x, (0, 2, 1)).reshape(batch_size * num_neurons, input_horizon)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.output_horizon)(x)
        return jnp.transpose(x.reshape(batch_size, num_neurons, self.output_horizon), (0, 2, 1))


class SharedMLPWrapper:
    """init/apply interface expected by the training loop; BatchNorm statistics are mutable, Dropout needs an RNG."""

    def __init__(self):
        self.model = SharedNeuronMLP()
        self.mutable = ["batch_stats"]
        self.needs_rng = True

    def init(self, rng_key, dummy_input):
        rng_params, rng_dropout = jax.random.split(rng_key)
        return self.model.init({"params": rng_params, "dropout": rng_dropout}, dummy_input, training=True)

    def apply(self, params, x, training=False, mutable=None, rngs=None):
        rngs = rngs if rngs is not None else {}
        if mutable is not None:
            return self.model.apply(params, x, training=training, mutable=mutable, rngs=rngs)
        return self.model.apply(params, x, training=training, rngs=rngs)


def create_network(hparams):
    return SharedMLPWrapper()


def compute_loss(predictions, targets, params, x):
    return jnp.mean(jnp.abs(predictions - targets))


def create_optimizer(learning_rate=0.001):
    return optax.adam(learning_rate=learning_rate, eps=1e-8)
# EVOLVE-BLOCK-END
