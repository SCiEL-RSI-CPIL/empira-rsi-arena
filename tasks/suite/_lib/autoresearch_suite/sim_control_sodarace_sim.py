# MIT License
#
# Copyright (c) 2022 CarperAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Sodarace simulator vendored from OpenELM (MIT, CarperAI).

Upstream: https://github.com/CarperAI/OpenELM at commit
c844e149e3f59fef546e0bc55f4e12e0f192feb9, files
src/openelm/environments/sodaracer/simulator.py (sha256 fb2f8a44...),
helpers.py (e5dab42b...) and walker/walk_creator.py (527beb4f...). OpenELM's
simulator is itself a Python translation of OptimusLime/iesor-physics (Box2D).

Changes for this suite (the physics is unchanged):

* removed the pygame/OpenCV drawing framework base class, the JSON export of the
  world and the unused polygon/rectangle entity branches (walkers only create
  circle nodes);
* the debug ``print`` before re-raising a muscle-loading error was dropped;
* OpenELM builds ``Muscle(id, joint, amplitude, phase)`` positionally while the
  dataclass declares ``(id, joint, phase, amplitude)``, so upstream muscles
  oscillate with the walker's phase as amplitude and vice versa. (The C++
  original makes the same positional swap but also overwrites the phase with
  the amplitude while loading, so there every spring keeps its amplitude and
  all springs share one phase; it also scales the rest length multiplicatively
  where OpenELM adds.) The default here assigns amplitude and phase by name and
  keeps OpenELM's additive rule; ``upstream_muscle_order=True`` restores
  OpenELM's behaviour and is used by the cross-check against the upstream code;
* ``SodaraceSimulator`` gained ``center_of_mass()`` and ``advance(updates)``,
  used by the evaluator to measure the centre-of-mass displacement that the ELM
  paper defines as fitness (OpenELM's own ``evaluate`` returns
  ``abs(min_x + offsetX)``, which is kept below for the cross-check only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from Box2D import Box2D as b2

# --------------------------------------------------------------------------- helpers.py


class Entity:
    @staticmethod
    def circle_entity(body_id: str, x: float, y: float, radius: float, angle: float) -> dict:
        return {"type": 0, "bodyID": body_id, "x": x, "y": y, "radius": radius, "angle": angle}


class DistanceAccessor(b2.b2DistanceJoint):
    """Helper class to get/set the length of a distance joint."""

    getLength = b2._swig_new_instance_method(b2._Box2D.b2DistanceJoint___GetLength)
    setLength = b2._swig_new_instance_method(b2._Box2D.b2DistanceJoint___SetLength)


@dataclass
class Muscle:
    id: str
    joint: b2.b2Joint
    phase: float
    amplitude: float

    def __post_init__(self):
        self.original_length: float = DistanceAccessor.getLength(self.joint)


@dataclass
class Bone:
    id: str
    joint: b2.b2Joint


# --------------------------------------------------------------------------- simulator.py


class IESoRWorld:
    """Class for the Sodarace simulation (OpenELM's IESoRWorld without the drawing framework)."""

    def __init__(self, canvas_size: tuple[int, int] = (150, 200), upstream_muscle_order: bool = False):
        self.upstream_muscle_order = upstream_muscle_order
        self.interpolation: float = 0.0
        self.radians: float = 0.0
        self.accumulator: float = 0.0
        self.desiredFPS: float = 60.0
        self.simulation_rate: float = 1.0 / self.desiredFPS
        self.static_body_count: int = 0
        self.body_list: list = []
        self.shape_list: list = []
        self.body_map: dict = {}
        self.bone_list: list = []
        self.muscle_list: list = []
        self.gravity = b2.b2Vec2(0.0, -25.0)
        self.world: b2.b2World = b2.b2World(self.gravity)
        self.world.autoClearForces = False
        self.groundBodyDef: b2.b2BodyDef = b2.b2BodyDef()
        self.groundBodyDef.type = b2.b2_staticBody
        self.groundBodyDef.position = b2.b2Vec2(0.0, -100.0)
        self.groundBody: b2.b2Body = self._add_body_to_world("ground", self.groundBodyDef)
        self.groundBox: b2.b2PolygonShape = b2.b2PolygonShape()
        self.groundBox.SetAsBox(3500.0, 10.0)
        self._add_shape_to_body(self.groundBody, self.groundBox)
        self.canvas: b2.b2Vec2 = b2.b2Vec2(canvas_size[0], canvas_size[1])

    def update_world(self, ms_update: float) -> int:
        """Simulate the world for a given amount of time (milliseconds); returns the number of physics steps."""
        ms_update *= 4  # as upstream
        step_count: int = 0
        frame_time: float = ms_update / 1000.0
        frame_time = 0.35 if frame_time > 0.35 else frame_time
        self.accumulator += frame_time
        while self.accumulator >= self.simulation_rate:
            step_count += 1
            speedup: float = 3.0
            for muscle in self.muscle_list:
                distance_joint: b2.b2Joint = muscle.joint
                length_calc = muscle.original_length + (muscle.amplitude * math.cos(self.radians + muscle.phase * 2 * math.pi))
                DistanceAccessor.setLength(distance_joint, length_calc)
            self.world.Step(self.simulation_rate, 10, 10)
            self.world.ClearForces()
            self.radians += speedup * self.simulation_rate
            self.accumulator -= self.simulation_rate
        self.interpolation = self.accumulator / self.simulation_rate
        return step_count

    def _add_body_to_world(self, body_id: str, body_def: b2.b2BodyDef) -> b2.b2Body:
        body: b2.b2Body = self.world.CreateBody(body_def)
        self.body_list.append(body_id)
        self.body_map[body_id] = body
        body.userData = body_id
        return body

    def _add_shape_to_body_fixture(self, body: b2.b2Body, fix_def: b2.b2FixtureDef) -> b2.b2Fixture:
        shape_id = self.create_shape_id()
        fixture: b2.b2Fixture = body.CreateFixture(fix_def)
        fixture.userData = shape_id
        return fixture

    def _add_shape_to_body(self, body: b2.b2Body, shape: b2.b2Shape, density: float = 0.0) -> b2.b2Fixture:
        shape_id = self.create_shape_id()
        fixture: b2.b2Fixture = body.CreateFixture(shape=shape, density=density)
        fixture.userData = shape_id
        return fixture

    def load_body_into_world(self, body: dict) -> dict:
        """Load a walker dict ("joints", "muscles") into the world; returns its morphology."""
        o_body_count = self.world.bodyCount
        morphology: dict[str, float] = {}
        entities: list[dict[str, Any]] = self._get_body_entities(body, morphology)
        entity_map: dict[str, dict] = {e["bodyID"]: e for e in entities}
        for entity in entities:
            self.set_body(entity)
        connections = body["muscles"]
        amplitude_cutoff = 0.2
        connection_distance_sum = 0.0
        for muscle in connections:
            source_id = str(o_body_count + muscle[0])
            target_id = str(o_body_count + muscle[1])
            if source_id == target_id:
                continue
            muscle_properties: dict = muscle[-1]
            amplitude = (muscle_properties["amplitude"] + 1) / 2.0  # [-1, 1] -> [0, 1] as upstream
            connection_distance = math.sqrt((entity_map[source_id]["x"] - entity_map[target_id]["x"]) ** 2
                                            + (entity_map[source_id]["y"] - entity_map[target_id]["y"]) ** 2)
            connection_distance_sum += connection_distance
            properties: dict[str, float] = {}
            if amplitude < amplitude_cutoff:
                self.add_distance_joint(source_id, target_id, properties)
            else:
                properties["frequencyHz"] = 3
                properties["dampingRatio"] = 0.3
                properties["phase"] = muscle_properties["phase"]
                properties["amplitude"] = 0.3 * amplitude
                self.add_muscle_joint(source_id, target_id, properties)
        morphology["mass"] = morphology["totalNodes"] + connection_distance_sum / 2.0
        del morphology["totalNodes"]
        return morphology

    def _get_body_entities(self, body: dict, initial_morphology: dict) -> list[dict[str, Any]]:
        entity_vector: list[dict[str, Any]] = []
        o_nodes: list = body["joints"]
        body_id: int = self.world.bodyCount
        divide_for_max_width: float = 2.2
        divide_for_max_height: float = 2.5
        max_allowed_width: float = self.canvas.x / divide_for_max_width
        max_allowed_height: float = self.canvas.y / divide_for_max_height
        min_x, max_x, min_y, max_y = self.canvas.x, 0.0, self.canvas.y, 0.0
        for node in o_nodes:
            node_x, node_y = node[0], node[1]
            x_scaled = node_x / divide_for_max_width * max_allowed_width
            y_scaled = (1.0 - node_y) / divide_for_max_height * max_allowed_height
            entity_vector.append(Entity.circle_entity(str(body_id), x_scaled, y_scaled, max_allowed_width / 35.0, 0))
            min_x = min(min_x, x_scaled)
            max_x = max(max_x, x_scaled)
            min_y = min(min_y, y_scaled)
            max_y = max(max_y, y_scaled)
            body_id += 1
        move_x: float = (max_x - min_x) / 2.0
        move_y: float = (max_y - min_y) / 2.0
        for entity in entity_vector:
            entity["x"] = (float(entity["x"]) - min_x) + self.canvas.x / 2 - move_x
            entity["y"] = (float(entity["y"]) - min_y) + self.canvas.y / 2 - move_y
        initial_morphology["width"] = max_x - min_x
        initial_morphology["height"] = max_y - min_y
        initial_morphology["startX"] = min_x
        initial_morphology["startY"] = min_y
        initial_morphology["offsetX"] = -min_x + self.canvas.x / 2 - move_x
        initial_morphology["offsetY"] = -min_y + self.canvas.y / 2 - move_y
        initial_morphology["totalNodes"] = len(o_nodes)
        return entity_vector

    def add_distance_joint(self, source_id: str, target_id: str, properties: dict) -> Bone:
        body1: b2.b2Body = self.body_map[source_id]
        body2: b2.b2Body = self.body_map[target_id]
        joint = b2.b2DistanceJointDef()
        joint.Initialize(body1, body2, body1.worldCenter, body2.worldCenter)
        if properties.get("frequencyHz", None) is not None:
            joint.frequencyHz = properties["frequencyHz"]
        if properties.get("dampingRatio", None) is not None:
            joint.dampingRatio = properties["dampingRatio"]
        w_joint = self.world.CreateJoint(joint)
        bone = Bone(str(len(self.bone_list)), w_joint)
        self.bone_list.append(bone)
        w_joint.userData = bone
        return bone

    def add_muscle_joint(self, source_id: str, target_id: str, properties: dict) -> Muscle:
        added_joint = self.add_distance_joint(source_id, target_id, properties)
        phase = 0 if properties.get("phase", None) is None else properties["phase"]
        amplitude = 1 if properties.get("amplitude", None) is None else properties["amplitude"]
        if self.upstream_muscle_order:
            # OpenELM passes (amplitude, phase) positionally into the fields (phase, amplitude), which swaps them
            muscle = Muscle(str(len(self.muscle_list)), added_joint.joint, amplitude, phase)
        else:
            muscle = Muscle(str(len(self.muscle_list)), added_joint.joint, phase=phase, amplitude=amplitude)
        self.muscle_list.append(muscle)
        return muscle

    def set_body(self, entity: dict) -> None:
        """Add a circle node (dynamic body, fixed rotation, linear damping 1.1) to the world."""
        body_def = b2.b2BodyDef()
        body_def.type = b2.b2_dynamicBody
        body_def.position = b2.b2Vec2(entity["x"], entity["y"])
        body_def.linearDamping = 1.1
        body_def.fixedRotation = True
        if entity.get("angle", None) is not None:
            body_def.angle = entity["angle"]
        body: b2.b2Body = self._add_body_to_world(entity["bodyID"], body_def)
        filter = b2.b2Filter()  # nodes collide with the ground only
        filter.categoryBits = 0x0002
        filter.maskBits = 0x0001
        filter.groupIndex = 0
        fixture = b2.b2FixtureDef()
        fixture.density = 25.0
        fixture.friction = 1.0
        fixture.restitution = 0.1
        node = b2.b2CircleShape()
        node.radius = entity["radius"]
        fixture.shape = node
        fixture.filter = filter
        self._add_shape_to_body_fixture(body, fixture)

    def create_shape_id(self) -> str:
        self.static_body_count += 1
        shape_id = str(self.static_body_count)
        self.shape_list.append(shape_id)
        return shape_id


class SodaraceSimulator:
    """Single-walker simulator (upstream interface plus centre-of-mass helpers)."""

    def __init__(self, body: dict, upstream_muscle_order: bool = False) -> None:
        self.world = IESoRWorld(upstream_muscle_order=upstream_muscle_order)
        self._morphology = self.world.load_body_into_world(body)
        self.nodes = [b for b in self.world.world.bodies if b.type == b2.b2_dynamicBody]

    @property
    def morphology(self) -> dict:
        return self._morphology

    def evaluate(self, time: float) -> float:
        """OpenELM's fitness (kept for the cross-check): abs(min bodyA x + offsetX) after time // 350 updates."""
        for _ in range(int(time // 350)):
            self.world.update_world(350)
        end = min([bone.joint.bodyA.position[0] for bone in self.world.bone_list]
                  + [muscle.joint.bodyA.position[0] for muscle in self.world.muscle_list])
        return abs(end + self.morphology["offsetX"])

    def center_of_mass(self) -> tuple[float, float]:
        """Mass-weighted mean position of the walker's nodes (local addition)."""
        total = sum(body.mass for body in self.nodes)
        x = sum(body.mass * body.worldCenter[0] for body in self.nodes) / total
        y = sum(body.mass * body.worldCenter[1] for body in self.nodes) / total
        return x, y

    def advance(self, updates: int, ms: float = 350.0) -> int:
        """Run ``updates`` calls of update_world(ms); returns the number of physics steps (local addition)."""
        return sum(self.world.update_world(ms) for _ in range(int(updates)))


# --------------------------------------------------------------------------- walker/walk_creator.py

WALKER_CREATOR_SOURCE = '''
import math as _wc_math
from itertools import combinations as _wc_combinations


class Walker:
    """Joints and muscles of a Sodaracer (OpenELM's Walker dataclass written as a plain class)."""

    def __init__(self, joints, muscles):
        self.joints = joints
        self.muscles = muscles

    def __repr__(self):
        return "Walker(joints=%r, muscles=%r)" % (self.joints, self.muscles)

    def __eq__(self, other):
        return isinstance(other, Walker) and (self.joints, self.muscles) == (other.joints, other.muscles)

    def to_dict(self):
        return self.__dict__

    def validate(self):
        """OpenELM's validity rules: muscle strength <= 10, <= 10 muscles per joint, joints >= 0.003 apart."""
        max_muscles_per_joint = 10
        max_muscle_strength = 10
        min_joint_distance = 0.003
        if len(self.joints) == 0 and len(self.muscles) == 0:
            return False
        for m in self.muscles:
            if m[-1]["type"] == "muscle":
                if m[-1]["amplitude"] > max_muscle_strength:
                    return False
        for j in self.joints:
            joint_idx = self.joints.index(j)
            count = 0
            for m in self.muscles:
                count += m[:2].count(joint_idx)
            if count > max_muscles_per_joint:
                return False
        for j1, j2 in _wc_combinations(self.joints, r=2):
            if _wc_math.sqrt(((j1[0] - j2[0]) ** 2 + (j1[1] - j2[1]) ** 2)) < min_joint_distance:
                return False
        return True


SCALE = 0.22


class walker_creator:
    """Walker creator of the ELM paper (OpenELM walk_creator.py)."""

    def __init__(self):
        self.joints = []
        self.muscles = []

    def add_joint(self, x, y):
        """Add a point mass (joint); returns its scaled position."""
        j = (x * SCALE, -y * SCALE)
        self.joints.append(j)
        return j

    def add_muscle(self, j0, j1, amplitude=0.0, phase=0.0):
        """Add a spring (muscle) between two joints returned by add_joint."""
        muscle_data = [self.joints.index(j0), self.joints.index(j1)]
        muscle_data.append({"type": "distance", "amplitude": amplitude, "phase": phase})
        if amplitude != 0.0 or phase != 0.0:
            muscle_data[2]["type"] = "muscle"
        self.muscles.append(muscle_data)
        return muscle_data

    def get_walker(self):
        return Walker(self.joints, self.muscles)
'''

_namespace: dict = {}
exec(compile(WALKER_CREATOR_SOURCE, "<walk_creator>", "exec"), _namespace)  # one source for evaluator and sandbox
Walker = _namespace["Walker"]
walker_creator = _namespace["walker_creator"]
SCALE = _namespace["SCALE"]
