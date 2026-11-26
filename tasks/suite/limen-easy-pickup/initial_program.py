"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp

# Tile ids: BALL 3, SQUARE 4, PYRAMID 5, HEX 11. Colour ids: GREEN 2, BLUE 3, YELLOW 5.
# Actions: 0 forward, 1 turn right, 2 turn left, 3 pick up, 4 put down, 5 toggle.


def get_observation(state):
    """LIMEN's default observation: grid tile and colour ids / 12, agent position, direction, pocket and step counter."""
    height, width = state.grid.shape[:2]
    size = max(height, width)
    grid = state.grid.reshape(-1).astype(jnp.float32) / 12.0
    position = state.agent.position.astype(jnp.float32) / (size - 1)
    direction = jax.nn.one_hot(state.agent.direction, 4)
    pocket = state.agent.pocket.astype(jnp.float32) / 12.0
    step = jnp.array([state.step_num.astype(jnp.float32) / 100.0])
    return jnp.concatenate([grid, position, direction, pocket, step])


def compute_reward(state, action, next_state):
    """Sparse task reward: 1 when the agent picks up the blue pyramid, otherwise 0."""
    holding = (next_state.agent.pocket[0] == 5) & (next_state.agent.pocket[1] == 3)
    return ((action == 3) & holding).astype(jnp.float32)
# EVOLVE-BLOCK-END
