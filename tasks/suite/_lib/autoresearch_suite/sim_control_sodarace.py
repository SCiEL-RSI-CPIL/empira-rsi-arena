"""Sodarace walker design (ELM): score = horizontal centre-of-mass displacement in the Box2D simulator.

The candidate is a walker program in the format of the ELM paper::

    def make_walker():
        wc = walker_creator()
        ...  # wc.add_joint(x, y) -> joint, wc.add_muscle(j0, j1, amplitude, phase)
        return wc.get_walker()

``walker_creator`` and ``Walker`` are provided by the evaluator (definitions in
the candidate are replaced). ``make_walker()`` runs in a resource-limited
sandboxed child (``software_sandbox``); only the resulting joint and muscle
lists come back. The evaluator validates them (OpenELM's validity rules plus
size caps), simulates the walker with the vendored OpenELM/iesor-physics Box2D
simulator (sim_control_sodarace_sim) for 80 calls of update_world(350)
(1,680 physics steps of 1/60 s, 28 s of simulated time) and returns the
absolute horizontal displacement of the walker's centre of mass, in simulator
units, which is how the ELM paper defines Sodarace fitness.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from . import sim_control_sodarace_sim as sim
from . import software_sandbox
from .runtime import CandidateError

ENTRY = "__sim_control_sodarace_walker__"
MAX_JOINTS = 100
MAX_MUSCLES = 500
MAX_SPAN = 30.0  # walker units after the creator's 0.22 scale; about 930 simulator units
MAX_ABS_COORDINATE = 1000.0
MAX_ABS_PHASE = 1000.0
MAX_MUSCLE_STRENGTH = 10.0  # OpenELM Walker.validate
MAX_MUSCLES_PER_JOINT = 10
MIN_JOINT_DISTANCE = 0.003
GROUND_FLOOR = -100.0  # centre line of the 20-unit-thick ground box (its top surface is at -90)

SHIM = '''


class SodaraceContractError(Exception):
    pass


def __sim_control_sodarace_walker__():
    import math as _math
    import numbers as _numbers
    import random as _random
    _random.seed(0)

    def real(value):  # booleans and strings (also numeric ones) are not coordinates, amplitudes or phases
        if isinstance(value, bool) or not isinstance(value, _numbers.Real):
            raise TypeError("not a real number")
        return float(value)

    try:
        import numpy as _np
        _np.random.seed(0)
    except ImportError:
        pass
    function = globals().get("make_walker")
    if not callable(function):
        raise SodaraceContractError("the candidate must define make_walker()")
    result = function()
    if not isinstance(result, Walker):
        raise SodaraceContractError("make_walker() must return wc.get_walker(), got " + type(result).__name__)
    joints, muscles = [], []
    for joint in list(result.joints):
        try:
            x, y = joint
            joints.append([real(x), real(y)])
        except (TypeError, ValueError):
            raise SodaraceContractError("every joint must be an (x, y) pair of numbers") from None
    for muscle in list(result.muscles):
        try:
            source, target, props = muscle
            if any(isinstance(index, bool) or not isinstance(index, _numbers.Integral) for index in (source, target)):
                raise TypeError("not a joint index")
            muscles.append([int(source), int(target), {"amplitude": real(props["amplitude"]), "phase": real(props["phase"]),
                                                       "type": str(props.get("type", ""))}])
        except (TypeError, ValueError, KeyError):
            raise SodaraceContractError("every muscle must be [joint_index, joint_index, {amplitude, phase}] as built by add_muscle") from None
    for value in [c for j in joints for c in j] + [m[2][k] for m in muscles for k in ("amplitude", "phase")]:
        if not _math.isfinite(value):
            raise SodaraceContractError("joint coordinates, amplitudes and phases must be finite")
    return {"joints": joints, "muscles": muscles}
'''


def program_source(candidate_source):
    """Candidate first (so it may start with __future__ imports), then the official creator and the driver."""
    return candidate_source + "\n\n" + sim.WALKER_CREATOR_SOURCE + SHIM


def check_source(module):
    path = software_sandbox.require_entry(module, "make_walker")
    source = Path(path).read_text(encoding="utf-8")
    if ENTRY in source:
        raise CandidateError(f"the name {ENTRY} is reserved")
    return source


def build_walker(source, cpu_seconds=30, memory_mb=1024):
    try:
        return software_sandbox.execute(program_source(source), ENTRY, [], cpu_seconds=cpu_seconds, memory_mb=memory_mb,
                                        filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        text = str(failure)
        contract = "SodaraceContractError" in text
        if contract:
            text = text.split("SodaraceContractError:", 1)[-1].strip()
        last = failure.traceback_text.strip().splitlines()[-1] if failure.traceback_text.strip() else ""
        detail = f" ({last})" if failure.kind == "exception" and not contract and last and last not in text else ""
        raise CandidateError(f"make_walker() failed ({failure.kind}): {text}{detail}"[:1500]) from None


def _number(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CandidateError(f"{what} must be a finite number")
    return float(value)


def validate(walker):
    """Independent check of the walker returned by the sandbox; returns the simulator body dict."""
    if not isinstance(walker, dict) or not isinstance(walker.get("joints"), list) or not isinstance(walker.get("muscles"), list):
        raise CandidateError("malformed walker")
    raw_joints, raw_muscles = walker["joints"], walker["muscles"]
    if not raw_joints:
        raise CandidateError("the walker has no joints")
    if len(raw_joints) > MAX_JOINTS or len(raw_muscles) > MAX_MUSCLES:
        raise CandidateError(f"at most {MAX_JOINTS} joints and {MAX_MUSCLES} muscles are allowed "
                             f"(got {len(raw_joints)} and {len(raw_muscles)})")
    joints = []
    for joint in raw_joints:
        if not isinstance(joint, list) or len(joint) != 2:
            raise CandidateError("every joint must be an (x, y) pair")
        x, y = (_number(c, "joint coordinates") for c in joint)
        if abs(x) > MAX_ABS_COORDINATE or abs(y) > MAX_ABS_COORDINATE:
            raise CandidateError(f"joint coordinates must lie within +-{MAX_ABS_COORDINATE:g}")
        joints.append((x, y))
    xs, ys = [j[0] for j in joints], [j[1] for j in joints]
    if max(xs) - min(xs) > MAX_SPAN or max(ys) - min(ys) > MAX_SPAN:
        raise CandidateError(f"the walker must fit in a {MAX_SPAN:g} x {MAX_SPAN:g} box of walker units "
                             "(add_joint scales coordinates by 0.22)")
    for a in range(len(joints)):
        for b in range(a + 1, len(joints)):
            if math.dist(joints[a], joints[b]) < MIN_JOINT_DISTANCE:
                raise CandidateError(f"joints {a} and {b} are closer than {MIN_JOINT_DISTANCE} (OpenELM validity rule)")
    muscles, per_joint = [], [0] * len(joints)
    for muscle in raw_muscles:
        if not isinstance(muscle, list) or len(muscle) != 3 or not isinstance(muscle[2], dict):
            raise CandidateError("every muscle must be [joint_index, joint_index, properties]")
        source, target = muscle[0], muscle[1]
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(joints) for i in (source, target)):
            raise CandidateError("muscle endpoints must be indices of existing joints")
        amplitude = _number(muscle[2].get("amplitude"), "muscle amplitudes")
        phase = _number(muscle[2].get("phase"), "muscle phases")
        if amplitude > MAX_MUSCLE_STRENGTH:
            raise CandidateError(f"muscle amplitudes must be at most {MAX_MUSCLE_STRENGTH:g} (OpenELM validity rule)")
        if abs(phase) > MAX_ABS_PHASE:
            raise CandidateError(f"muscle phases must lie within +-{MAX_ABS_PHASE:g}")
        per_joint[source] += 1
        per_joint[target] += 1
        muscles.append([source, target, {"type": "muscle" if amplitude != 0.0 or phase != 0.0 else "distance",
                                         "amplitude": amplitude, "phase": phase}])
    if max(per_joint) > MAX_MUSCLES_PER_JOINT:
        raise CandidateError(f"a joint may be connected to at most {MAX_MUSCLES_PER_JOINT} muscles (OpenELM validity rule)")
    return {"joints": joints, "muscles": muscles}


def simulate(body, updates=80, ms=350.0, max_seconds=120.0):
    """Centre-of-mass trajectory of the walker; raises CandidateError on a physically invalid run."""
    started = time.time()
    simulator = sim.SodaraceSimulator(body)
    x0, y0 = simulator.center_of_mass()
    steps = 0
    for update in range(int(updates)):
        steps += simulator.world.update_world(ms)
        positions = [(node.position[0], node.position[1]) for node in simulator.nodes]
        if not all(math.isfinite(c) for p in positions for c in p):
            raise CandidateError("the simulation produced non-finite positions")
        if min(p[1] for p in positions) < GROUND_FLOOR:
            raise CandidateError("a node passed through the ground (physically invalid run)")
        if time.time() - started > max_seconds:
            raise CandidateError(f"the simulation exceeded {max_seconds:g} s")
    x1, y1 = simulator.center_of_mass()
    return {"dx": x1 - x0, "start": (x0, y0), "end": (x1, y1), "steps": steps, "morphology": simulator.morphology,
            "seconds": time.time() - started}


def score_walker(walker, P):
    body = validate(walker)
    run = simulate(body, updates=int(P.get("updates", 80)), ms=float(P.get("update_ms", 350.0)))
    morphology = run["morphology"]
    details = {"joints": len(body["joints"]), "muscles": len(body["muscles"]), "signed_displacement": round(run["dx"], 6),
               "final_center_of_mass": [round(c, 4) for c in run["end"]], "physics_steps": run["steps"],
               "morphology": {k: round(float(morphology[k]), 4) for k in ("width", "height", "mass")},
               "simulation_seconds": round(run["seconds"], 3)}
    return abs(run["dx"]), details


def score(module, P):
    """Family entry point: |centre-of-mass x displacement| of the candidate's walker (higher is better)."""
    walker = build_walker(check_source(module), cpu_seconds=int(P.get("cpu_seconds", 30)))
    return score_walker(walker, P)
