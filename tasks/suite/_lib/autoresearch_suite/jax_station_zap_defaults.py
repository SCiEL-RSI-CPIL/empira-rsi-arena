"""Defaults of the Station's ZAPBench task, used when a candidate omits a hook.

Port of example/research_misc/zapbench/research/storage/system/defaults.py from
dualverse-ai/station@782088e5 (Apache-2.0): a shared-neuron MLP (every neuron's
4-step context -> Dense(64) -> BatchNorm -> ReLU -> Dropout(0.1) -> Dense(32),
weights shared across all 71,721 neurons), mean-absolute-error loss and Adam
(eps 1e-8). The Station's default BASE_SEED is time-based; this task fixes the
seed instead (see jax_station_zap_train.py). Imported by path in the training
child; it must not import the evaluator package.
"""
import flax.linen as nn
import jax.numpy as jnp
import optax
from jax import random


def default_define_hyperparameters():
    return {"learning_rate": 0.001}


class SharedNeuronMLP(nn.Module):
    hidden_size: int = 64
    output_horizon: int = 32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        batch_size, input_horizon, num_neurons = x.shape
        x = jnp.transpose(x, (0, 2, 1))
        x = jnp.reshape(x, (batch_size * num_neurons, input_horizon))
        x = nn.Dense(self.hidden_size)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.output_horizon)(x)
        x = jnp.reshape(x, (batch_size, num_neurons, self.output_horizon))
        return jnp.transpose(x, (0, 2, 1))


class SharedMLPWrapper:
    def __init__(self):
        self.model = SharedNeuronMLP()
        self.mutable = ["batch_stats"]
        self.needs_rng = True

    def init(self, rng_key, dummy_input):
        rng_params, rng_dropout = random.split(rng_key)
        return self.model.init({"params": rng_params, "dropout": rng_dropout}, dummy_input, training=True)

    def apply(self, params, x, training=False, mutable=None, rngs=None):
        if mutable is not None:
            return self.model.apply(params, x, training=training, mutable=mutable, rngs=rngs if rngs is not None else {})
        return self.model.apply(params, x, training=training, rngs=rngs if rngs is not None else {})


def default_create_network(hparams):
    return SharedMLPWrapper()


def default_compute_loss(predictions, targets, params, x):
    return jnp.mean(jnp.abs(predictions - targets))


def default_create_optimizer(learning_rate=0.001):
    return optax.adam(learning_rate=learning_rate, eps=1e-8)


DEFAULTS = {
    "_define_hyperparameters": default_define_hyperparameters,
    "create_network": default_create_network,
    "compute_loss": default_compute_loss,
    "create_optimizer": default_create_optimizer,
}
