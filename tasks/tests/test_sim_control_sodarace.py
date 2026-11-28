"""Tests of the Sodarace walker evaluator and the vendored OpenELM simulator (sim_control_sodarace*).

Ported from the source repository. Changes: the seed is read from the generated package (the generator module
scripts/families_sim_control.py is not part of this library); the optional OpenELM checkout is looked up in the task
cache; the seed's recorded score is compared per platform, because Box2D is single precision and its builds do not
round alike (SEED_SCORES below).
"""
import importlib
import platform
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from arena_task_paths import cache_path  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

try:
    import Box2D  # noqa: F401
    HAVE_BOX2D = True
except ImportError:  # pragma: no cover
    HAVE_BOX2D = False

from autoresearch_suite.runtime import CandidateError  # noqa: E402

UPSTREAM = cache_path("sim_control-raw", "openelm", "src", "openelm", "environments", "sodaracer")
SEED_TASK = "elm-sodarace-walker-program-evolution"
# Centre-of-mass displacement of the seed after 1,680 Box2D steps. The source repository recorded it on macOS arm64
# (validation/sim_control-checks.json); the Linux x86_64 wheel of the same Box2D 2.3.10 ends 2.6e-5 away, about the
# size of Box2D's single-precision rounding along this trajectory (a 1e-6 shift of one joint moves the score by
# 4.7e-5). OpenELM's own two-update fitness of the square seed (81.8181828585538) is identical on both.
SEED_SCORES = {("Darwin", "arm64"): 0.7898643493652315, ("Linux", "x86_64"): 0.7898384094238224}
SEED_SCORE = SEED_SCORES.get((platform.system(), platform.machine()))


def assert_seed_score(test, value):
    """The recorded value to 6 places where one is recorded for the platform, otherwise within the platform spread."""
    if SEED_SCORE is None:
        test.assertAlmostEqual(value, 0.78985, delta=5e-4)
    else:
        test.assertAlmostEqual(value, SEED_SCORE, places=6)
SQUARE_BODY = {"joints": [(0.0, 0.0), (0.0, -2.2), (2.2, -2.2), (2.2, 0.0), (1.1, -1.1)],
               "muscles": [[0, 1, {"type": "distance", "amplitude": 0.0, "phase": 0.0}], [1, 2, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
                           [2, 3, {"type": "distance", "amplitude": 0.0, "phase": 0.0}], [3, 0, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
                           [3, 4, {"type": "distance", "amplitude": 0.0, "phase": 0.0}], [0, 4, {"type": "muscle", "amplitude": 5.0, "phase": 0.0}],
                           [1, 4, {"type": "muscle", "amplitude": 10.0, "phase": 0.0}], [2, 4, {"type": "muscle", "amplitude": 2.0, "phase": 0.0}]]}


@unittest.skipUnless(HAVE_BOX2D, "Box2D is not installed")
class SimulatorTests(unittest.TestCase):
    def setUp(self):
        from autoresearch_suite import sim_control_sodarace_sim as sim
        self.sim = sim

    def test_walker_creator_matches_openelm(self):
        namespace = {"walker_creator": self.sim.walker_creator}
        exec(seed_source(SEED_TASK), namespace)
        walker = namespace["make_walker"]()
        self.assertEqual(walker.to_dict(), SQUARE_BODY)  # OpenELM's tests/test_sodarace_walkers.py dictionary
        self.assertTrue(walker.validate())

    def test_openelm_fitness_golden_value(self):
        # OpenELM's own fitness of its square seed, measured with the upstream code (validation/sim_control-checks.json).
        value = self.sim.SodaraceSimulator(SQUARE_BODY, upstream_muscle_order=True).evaluate(1000)
        self.assertAlmostEqual(value, 81.8181828585538, places=9)

    def test_named_muscle_order_differs_only_for_oscillating_springs(self):
        bones = {"joints": SQUARE_BODY["joints"], "muscles": [[0, 1, {"type": "muscle", "amplitude": -1.0, "phase": 0.7}],
                                                              [1, 2, {"type": "muscle", "amplitude": -1.0, "phase": 0.2}]]}
        a, b = self.sim.SodaraceSimulator(bones, True), self.sim.SodaraceSimulator(bones, False)
        a.advance(10)
        b.advance(10)
        self.assertEqual(a.center_of_mass(), b.center_of_mass())  # amplitude < -0.6: rigid bones, no muscle at all
        muscle = self.sim.SodaraceSimulator(SQUARE_BODY).world.muscle_list[-1]
        self.assertAlmostEqual(muscle.amplitude, 0.3 * (2.0 + 1) / 2)
        self.assertEqual(muscle.phase, 0.0)

    @unittest.skipUnless((UPSTREAM / "simulator.py").is_file(), "OpenELM checkout not prepared in <task cache>/sim_control-raw")
    def test_port_reproduces_the_upstream_simulator(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "openelm" / "environments" / "sodaracer"
            (package / "box2d_examples").mkdir(parents=True)
            (package / "walker").mkdir()
            for init in (Path(tmp) / "openelm", Path(tmp) / "openelm" / "environments", package, package / "box2d_examples"):
                (init / "__init__.py").write_text("")
            (package / "box2d_examples" / "framework.py").write_text("class Framework:\n    pass\n\n\ndef main(*args):\n    pass\n")
            for name in ("simulator.py", "helpers.py"):
                shutil.copy(UPSTREAM / name, package / name)
            for name in ("walk_creator.py", "radial.py", "CPPN_mutable.py"):
                shutil.copy(UPSTREAM / "walker" / name, package / "walker" / name)
            (package / "walker" / "__init__.py").write_text("from openelm.environments.sodaracer.walker.CPPN_mutable import query_cppn\n"
                                                             "from openelm.environments.sodaracer.walker.walk_creator import Walker, walker_creator\n")
            sys.path.insert(0, tmp)
            try:
                upstream = importlib.import_module("openelm.environments.sodaracer.simulator")
                walkers = [importlib.import_module("openelm.environments.sodaracer.walker.radial").make_walker(),
                           importlib.import_module("openelm.environments.sodaracer.walker.CPPN_mutable").make_walker()]
                for walker in walkers:
                    ours = self.sim.SodaraceSimulator(walker.to_dict(), upstream_muscle_order=True)
                    theirs = upstream.SodaraceSimulator(body=walker.to_dict())
                    for _ in range(20):
                        ours.world.update_world(350)
                        theirs.world.update_world(350)
                    mine = sorted((b.userData, tuple(b.position)) for b in ours.world.world.bodies)
                    other = sorted((b.userData, tuple(b.position)) for b in theirs.world.world.bodies)
                    self.assertEqual(mine, other)
            finally:
                sys.path.remove(tmp)
                for name in [m for m in sys.modules if m == "openelm" or m.startswith("openelm.")]:
                    del sys.modules[name]


@unittest.skipUnless(HAVE_BOX2D, "Box2D is not installed")
class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        from autoresearch_suite import sim_control_sodarace as S
        self.S = S
        self.seed = seed_source(SEED_TASK)

    def score_source(self, source):
        return self.S.score_walker(self.S.build_walker(source), {})

    def test_seed_scores_its_recorded_value(self):
        value, details = self.score_source(self.seed)
        assert_seed_score(self, value)
        self.assertEqual(details["physics_steps"], 1680)
        self.assertEqual((details["joints"], details["muscles"]), (5, 8))

    def test_future_import_and_creator_redefinition(self):
        value, _ = self.score_source("from __future__ import annotations\n\n" + self.seed + "\n\nclass walker_creator:\n    pass\n")
        assert_seed_score(self, value)

    def test_invalid_walkers_are_rejected(self):
        bad = {
            "strength": self.seed.replace("10.0, 0.0", "11.0, 0.0"),
            "dictionary": self.seed.replace("return wc.get_walker()", "return wc.get_walker().to_dict()"),
            "overlap": self.seed.replace("centre = wc.add_joint(5, 5)", "centre = wc.add_joint(0, 0.001)"),
            "nan": self.seed.replace("wc.add_joint(5, 5)", "wc.add_joint(float('nan'), 5)"),
            "span": self.seed.replace("make_square(wc, 0, 0, 10, 10)", "make_square(wc, 0, 0, 200, 10)"),
            "raises": self.seed.replace("wc = walker_creator()", "raise RuntimeError('no walker')"),
            "file": self.seed.replace("wc = walker_creator()", "open('/etc/hosts').read()\n    wc = walker_creator()"),
        }
        for name, source in bad.items():
            with self.subTest(name=name), self.assertRaises(CandidateError):
                self.score_source(source)

    def test_validation_of_raw_walkers(self):
        good = {"joints": [[0.0, 0.0], [1.0, 0.0]], "muscles": [[0, 1, {"amplitude": 1.0, "phase": 0.5, "type": "muscle"}]]}
        self.S.validate(good)
        cases = [
            {"joints": [[0.0, 0.0]] * 0, "muscles": []},
            {"joints": [[float(i), 0.0] for i in range(101)], "muscles": []},
            {"joints": [[0.0, 0.0], [1.0, 0.0]], "muscles": [[0, 2, {"amplitude": 1.0, "phase": 0.0}]]},
            {"joints": [[0.0, 0.0], [1.0, 0.0]], "muscles": [[0, 1, {"amplitude": 50.0, "phase": 0.0, "type": "distance"}]]},
            {"joints": [[0.0, 0.0], [1.0, 0.0]], "muscles": [[0, 1, {"amplitude": 1.0, "phase": 5000.0}]]},
            {"joints": [[0.0, 0.0], [1.0, 0.0]], "muscles": [[0, 1, {"amplitude": True, "phase": 0.0}]]},
            {"joints": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], "muscles": [[0, 1, {"amplitude": 1.0, "phase": 0.0}]] * 11},
        ]
        for walker in cases:
            with self.assertRaises(CandidateError):
                self.S.validate(walker)

    def test_family_entry_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.py"
            path.write_text(self.seed)
            module = types.SimpleNamespace(make_walker=lambda: None, __file__=str(path))
            value, _ = self.S.score(module, {"updates": 80})
            assert_seed_score(self, value)


if __name__ == "__main__":
    unittest.main()
