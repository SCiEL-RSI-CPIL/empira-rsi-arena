"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp

# Tile ids: BALL 3, SQUARE 4, PYRAMID 5, HEX 11. Colour ids: GREEN 2, BLUE 3, YELLOW 5.
# Actions: 0 forward, 1 turn right, 2 turn left, 3 pick up, 4 put down, 5 toggle.

DIRECTIONS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])


def put_down_completes(next_state, action, tile_a, tile_b):
    """XLand-MiniGrid's TileNearGoal check: after a put-down (action 4), the cell in front of the agent holds
    tile_a next to tile_b or tile_b next to tile_a (tiles are [tile id, colour id] pairs)."""
    grid = next_state.grid
    front = next_state.agent.position.astype(jnp.int32) + DIRECTIONS[next_state.agent.direction]
    padded = jnp.pad(grid, ((1, 1), (1, 1), (0, 0)))
    y, x = front[0] + 1, front[1] + 1
    neighbours = jnp.stack([padded[y - 1, x], padded[y, x + 1], padded[y + 1, x], padded[y, x - 1]])
    tile = grid[front[0], front[1]]
    a, b = jnp.array(tile_a), jnp.array(tile_b)
    near_a = jnp.any(jnp.all(neighbours == a, axis=-1))
    near_b = jnp.any(jnp.all(neighbours == b, axis=-1))
    return (action == 4) & ((jnp.all(tile == a) & near_b) | (jnp.all(tile == b) & near_a))


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
    """Sparse task reward: 1 when a put-down completes the goal (yellow pyramid next to green square), otherwise 0."""
    return put_down_completes(next_state, action, (5, 5), (4, 2)).astype(jnp.float32)
# EVOLVE-BLOCK-END
