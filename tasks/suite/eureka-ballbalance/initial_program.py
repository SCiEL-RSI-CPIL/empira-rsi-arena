"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def reward(observation, action, next_observation, info):
    # Sparse baseline: one unit for every step on which the ball is still on the tray
    # (the task fitness is the episode length). Shape it to train better policies.
    return 1.0
# EVOLVE-BLOCK-END
