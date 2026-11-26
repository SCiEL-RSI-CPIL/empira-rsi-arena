"""Seed program and contract text of the jax_station ZAPBench family (the source repository's generator writes them into the package)."""

SEED = '''"""Shared-neuron MLP: the Station's ZAPBench baseline (its system defaults, written out)."""
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
'''

CONTRACT = (
    "The Station's ZAPBench hooks, each optional (an omitted hook falls back to the Station's default: shared-neuron MLP, "
    "mean-absolute-error loss, Adam with learning rate 0.001, batch size 8): create_network(hparams) -> an object with "
    "init(rng_key, dummy_input) -> parameter pytree (dummy_input is float32 [1, 4, 71721]) and apply(params, x, "
    "training=False, mutable=None, rngs=None) -> float predictions [batch, 32, 71721] for x float32 [batch, 4, 71721]; "
    "optional attributes mutable (list of mutable Flax collections such as ['batch_stats']; apply then returns "
    "(predictions, updates) when called with mutable) and needs_rng (True to receive rngs={'dropout': key} in training); "
    "compute_loss(predictions, targets, params, x) -> scalar training loss; create_optimizer(learning_rate) -> "
    "optax.GradientTransformation (applied to params['params'] for Flax-style parameters, as in the Station); "
    "_define_hyperparameters() -> dict of plain JSON values (learning_rate is passed to create_optimizer); BATCH_SIZE "
    "int in 1..64. Data: ZAPBench 240930_traces (calcium activity of 71,721 neurons of a larval zebrafish, values about "
    "-0.25..1.5) with ZAPBench's condition-aware splits for a 4-step context: training windows (4 context + 32 target "
    "steps, stride 1, never crossing a condition) from the first 70 % of the 8 training conditions (4,772 windows), "
    "validation windows from the next 10 % (469 windows). Training: the Station's loop with one fixed seed (42; the "
    "candidate's BASE_SEED is ignored): shuffled epochs, the Station's train step, validation MAE after every epoch, "
    "early stopping with patience 20 and the best-validation parameters kept; reduced CPU budget of at most <EPOCHS> "
    "epochs and <CAP> s of training wall time with 4 threads (training stops before an epoch that is predicted to "
    "exceed the cap). Predictions are made one window at a time (apply on a single window with training=False, "
    "vmapped), exported with jax.export (host callbacks and foreign custom calls are rejected) and run on the test "
    "split by a separate process that never executes candidate code. Score (minimize): ZAPBench test MAE = mean over the "
    "9 test conditions (the last 20 % of each training condition plus the held-out taxis condition, condition 3, from "
    "252 steps after its start, as in ZAPBench's test_holdout split for a 4-step context) of the mean absolute error over "
    "all stride-1 test windows (1,557 in all), the 32 predicted steps and the 71,721 neurons")


def contract(epochs, cap_seconds):
    return CONTRACT.replace("<EPOCHS>", str(epochs)).replace("<CAP>", str(cap_seconds))
