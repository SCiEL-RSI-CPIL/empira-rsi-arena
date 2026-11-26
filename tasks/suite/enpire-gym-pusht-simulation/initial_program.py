"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

import numpy as np

GOAL = (256.0, 256.0, math.pi / 4)   # fixed goal pose of the T (x, y, angle)
COG = np.array([0.0, 45.0])          # centre of gravity of the T in its body frame
INERTIA = 3000.0                     # moment of inertia of the T (mass 1)
# Outline of the T in its body frame (y up): bar [-60, 60] x [0, 30], stem [-15, 15] x [30, 120].
OUTLINE = np.array([(-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30)], dtype=float)
RADIUS = 15.0                        # radius of the pushing disc
ANGLE_WEIGHT = 55.0                  # pixels per radian when comparing pose errors


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def contact_points():
    """Points sampled along the outline with their outward normals (body frame)."""
    points = []
    for k in range(len(OUTLINE)):
        a, b = OUTLINE[k], OUTLINE[(k + 1) % len(OUTLINE)]
        d = b - a
        length = float(np.hypot(*d))
        normal = np.array([d[1], -d[0]]) / length
        for f in np.linspace(0.2, 0.8, max(2, int(length // 20))):
            points.append((a + f * d, normal))
    return points


CONTACTS = contact_points()


def segment_distance(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.hypot(*(p - a - t * ab)))


def clearance(p, polygon):
    return min(segment_distance(p, polygon[k], polygon[(k + 1) % len(polygon)]) for k in range(len(polygon)))


def limit(agent, target, step=50.0):
    d = target - agent
    n = float(np.hypot(*d))
    return target if n <= step else agent + d * step / n


class Policy:
    """Greedy pusher: pick the contact and push depth that best reduce the pose error, walk around the T, push."""

    def __init__(self):
        self.plan = None
        self.phase = "plan"
        self.pushes = 0
        self.target = None

    def act(self, observation):
        agent = np.array(observation[:2], dtype=float)
        position, theta = np.array(observation[2:4], dtype=float), float(observation[4])
        R = rot(theta)
        polygon = position + OUTLINE @ R.T
        centre = position + R @ COG
        goal_centre = np.array(GOAL[:2]) + rot(GOAL[2]) @ COG
        error, angle_error = goal_centre - centre, wrap(GOAL[2] - theta)
        if self.phase == "plan":
            self.plan = self.choose(R, centre, error, angle_error)
            self.phase, self.pushes = "approach", 0
        point, normal, depth = self.plan
        q, n = position + R @ point, R @ normal
        start = q + n * (RADIUS + 5)
        if self.phase == "approach":
            if np.hypot(*(agent - start)) < 3.0:
                self.phase = "push"
                self.target = q + n * (RADIUS - depth)  # fixed in the world, so the T moves about `depth`
            else:
                return tuple(np.clip(self.navigate(agent, start, polygon, centre), 10, 502))
        self.pushes += 1
        if self.pushes > 6 or np.hypot(*(agent - self.target)) < 1.5:
            self.phase = "plan"
        return tuple(np.clip(self.target, 10, 502))

    def navigate(self, agent, target, polygon, centre):
        clear = all(clearance(agent + (target - agent) * t / 10, polygon) > RADIUS + 2 for t in range(1, 11))
        if clear:
            return limit(agent, target)
        here = agent - centre
        radius = float(np.hypot(*here))
        if clearance(agent, polygon) < RADIUS + 8 or radius < 95:
            return limit(agent, centre + here / max(radius, 1e-6) * max(radius + 30, 100))
        a0 = math.atan2(here[1], here[0])
        a1 = math.atan2(*(target - centre)[::-1])
        turn = wrap(a1 - a0)
        a = a0 + math.copysign(min(abs(turn), 0.5), turn)
        return limit(agent, centre + max(radius, 100) * np.array([math.cos(a), math.sin(a)]))

    def choose(self, R, centre, error, angle_error):
        """One-step lookahead with a frictionless point-push model (translation and rotation about the centre)."""
        best, best_cost = None, float(error @ error) + (ANGLE_WEIGHT * angle_error) ** 2
        for point, normal in CONTACTS:
            q, u = R @ point, -(R @ normal)
            r = q - R @ COG
            lever = r[0] * u[1] - r[1] * u[0]
            start = centre + r - u * (RADIUS + 5)
            if not (20 < start[0] < 492 and 20 < start[1] < 492):
                continue
            for depth in (1.0, 3.0, 8.0, 20.0, 40.0):
                shift = depth * u / (1.0 + lever * lever / INERTIA)
                turn = depth * lever / (INERTIA + lever * lever)
                cost = float((error - shift) @ (error - shift)) + (ANGLE_WEIGHT * (angle_error - turn)) ** 2
                if cost < best_cost:
                    best, best_cost = (point, normal, depth), cost
        return best if best is not None else (CONTACTS[0][0], CONTACTS[0][1], 2.0)
# EVOLVE-BLOCK-END
