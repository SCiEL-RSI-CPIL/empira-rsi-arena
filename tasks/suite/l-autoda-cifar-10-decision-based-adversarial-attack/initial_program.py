"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    """Seed: one Boundary-Attack-style move, a random step orthogonal to the line to the original, then a step towards it."""
    step = float(hyperparams[0])
    to_original = org_img.astype(np.float64) - best_adv_img
    distance = np.linalg.norm(to_original)
    if distance == 0.0:
        return best_adv_img
    direction = to_original / distance
    noise = std_normal_noise - np.vdot(std_normal_noise, direction) * direction
    noise /= np.linalg.norm(noise) + 1e-12
    candidate = best_adv_img + step * distance * noise
    return candidate + step * (org_img - candidate)
# EVOLVE-BLOCK-END
