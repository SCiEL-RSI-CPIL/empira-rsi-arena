"""Metrics, seeds and data derivation of the kaggle package on tiny synthetic data (no downloads)."""
from __future__ import annotations

import importlib.util
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import kaggle_data, kaggle_sandbox, runtime, sandboxed  # noqa: E402
from suite_test_support import load, package  # noqa: E402

# Requirements of the packages: pandas (evaluators, child process, derivation), scikit-learn and scipy (metrics, seeds).
MISSING = [name for name in ("pandas", "sklearn", "scipy") if importlib.util.find_spec(name) is None]
needs_packages = unittest.skipIf(MISSING, f"not installed: {', '.join(MISSING)}")
if not MISSING:
    import pandas as pd  # noqa: E402
    from autoresearch_suite import kaggle_derive, kaggle_derive_images, kaggle_tasks  # noqa: E402

RNG = np.random.RandomState(7)
CLASSES = [] if MISSING else kaggle_derive.JIGSAW_CLASSES
# The source repository reads seeds, registrations and labels from its generator (scripts/families_kaggle.py),
# which is not part of this library; the same values are read from the generated packages, keyed by the family
# each package states.
PACKAGES = {spec["family"]: spec for spec in map(package, load("catalog/upgrades/kaggle.json")["tasks"])}
LABELS = {spec["family"]: load(f"suite/{spec['id']}/problem.json")["labels"] for spec in PACKAGES.values()}


def write(root, task, files):
    """files: {(public|private, name): bytes}"""
    for (side, name), payload in files.items():
        path = root / task / side / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def fake_jigsaw(root, n=60, m=40):
    words = ["good", "nice", "idiot", "stupid", "hate", "kill", "fine", "thanks"]
    def comment(i):
        return " ".join(words[(i * k) % len(words)] for k in range(1, 6))
    train = [[f"t{i}", comment(i), *[int((i + j) % (j + 2) == 0) for j in range(6)]] for i in range(n)]
    test = [[f"s{i}", comment(i + 3)] for i in range(m)]
    labels = [[f"s{i}", *[int((i + j) % (j + 2) == 0) for j in range(6)]] for i in range(m)]
    write(root, "jigsaw", {("public", "train.csv"): kaggle_derive.csv_bytes(["id", "comment_text", *CLASSES], train),
                           ("public", "test.csv"): kaggle_derive.csv_bytes(["id", "comment_text"], test),
                           ("private", "test_labels.csv"): kaggle_derive.csv_bytes(["id", *CLASSES], labels)})
    return np.array([row[1:] for row in labels], dtype=float)


def fake_noisyoffice(root):
    clean = [np.full((6, 9), 255, np.uint8) for _ in range(5)]
    for k, image in enumerate(clean):
        image[2:4, k:k + 3] = 10
    noisy = [np.clip(c.astype(int) - 40 + RNG.randint(0, 30, c.shape), 0, 255).astype(np.uint8) for c in clean]
    bundle = lambda arrays: kaggle_derive.npz_bytes({f"a{i}": a for i, a in enumerate(arrays)})  # noqa: E731
    write(root, "noisyoffice", {("public", "train_noisy.npz"): bundle(noisy[:3]), ("public", "train_clean.npz"): bundle(clean[:3]),
                                ("public", "test_noisy.npz"): bundle(noisy[3:]), ("private", "test_clean.npz"): bundle(clean[3:])})
    return clean[3:]


def fake_pcam(root, n=80, m=40):
    labels = RNG.randint(0, 2, n + m)
    images = RNG.randint(0, 255, (n + m, 48, 48, 3)).astype(np.uint8)
    images[labels == 1, 16:32, 16:32, :] //= 3
    write(root, "pcam", {("public", "train.npz"): kaggle_derive.npz_bytes({"images": images[:n], "labels": labels[:n]}),
                         ("public", "test.npz"): kaggle_derive.npz_bytes({"images": images[n:]}),
                         ("private", "test_labels.npz"): kaggle_derive.npz_bytes({"labels": labels[n:]})})
    return labels[n:]


def fake_nomad(root, n=40, m=10):
    records = []
    for i in range(n + m):
        a = 9.0 + 0.1 * (i % 7)
        cell = np.diag([a, a + 0.5, a + 1.0])
        numbers = [13] * (i % 4 + 1) + [31] * (4 - i % 4) + [8] * 6 + [49] * 0
        record = kaggle_derive.nomad_record(cell, RNG.rand(len(numbers), 3) * a, numbers, [12, 33, 167][i % 3])
        record.update(id=i + 1, formation_energy_ev_natom=0.05 + 0.02 * (i % 5), bandgap_energy_ev=1.0 + 0.3 * (i % 4))
        records.append(record)
    train, test = records[:n], records[n:]
    targets = kaggle_derive.NOMAD_TARGETS
    write(root, "nomad", {("public", "train.jsonl"): kaggle_derive.jsonl_bytes(train),
                          ("public", "test.jsonl"): kaggle_derive.jsonl_bytes([{k: v for k, v in r.items() if k not in targets} for r in test]),
                          ("private", "test_labels.csv"): kaggle_derive.csv_bytes(["id", *targets], [[r["id"], r[targets[0]], r[targets[1]]] for r in test])})
    return np.array([[r[t] for t in targets] for r in test])


def fake_openvaccine(root, n=12, m=4, length=107, scored=68):
    def row(i, with_targets):
        seq = "".join("ACGU"[(i + k) % 4] for k in range(length))
        record = {"index": i, "id": f"id_{i:04d}", "sequence": seq, "structure": "." * length,
                  "predicted_loop_type": "".join("EHS"[(i * k) % 3] for k in range(length)), "seq_length": length, "seq_scored": scored}
        if with_targets:
            record.update(signal_to_noise=5.0, SN_filter=1)
            for column in kaggle_derive.OV_LIST_COLUMNS:
                record[column] = [round(float(x), 4) for x in RNG.rand(scored)]
        return record
    train = [row(i, True) for i in range(n)]
    test = [row(n + i, True) for i in range(m)]
    truth = np.stack([np.array([r[c] for c in kaggle_derive.OV_SCORED]).T for r in test])
    write(root, "openvaccine", {("public", "train.jsonl"): kaggle_derive.jsonl_bytes(train),
                                ("public", "test.jsonl"): kaggle_derive.jsonl_bytes([{k: r[k] for k in kaggle_derive.OV_TEST_COLUMNS} for r in test]),
                                ("private", "test_labels.npz"): kaggle_derive.npz_bytes({"targets": truth})})
    return truth


def fake_patent(root):
    train = [[f"tr{i}", f"anchor{i % 5}", f"anchor{i % 5}{'x' * (i % 3)}", "H01", [1.0, 0.5, 0.25][i % 3]] for i in range(30)]
    test = [[f"te{i}", f"query{i}", f"query{i}{'y' * (i % 3)}", "A01"] for i in range(9)]
    scores = [[f"te{i}", [1.0, 0.5, 0.25][i % 3]] for i in range(9)]
    write(root, "patent", {("public", "train.csv"): kaggle_derive.csv_bytes(["id", "anchor", "target", "context", "score"], train),
                           ("public", "test.csv"): kaggle_derive.csv_bytes(["id", "anchor", "target", "context"], test),
                           ("private", "test_labels.csv"): kaggle_derive.csv_bytes(["id", "score"], scores)})
    return np.array([s for _, s in scores])


FAKES = {"jigsaw": fake_jigsaw, "noisyoffice": fake_noisyoffice, "pcam": fake_pcam, "nomad": fake_nomad,
         "openvaccine": fake_openvaccine, "patent": fake_patent}
SEEDS = {"jigsaw": PACKAGES["kaggle_jigsaw_toxic"]["seed"], "noisyoffice": PACKAGES["kaggle_noisyoffice_denoising"]["seed"],
         "pcam": PACKAGES["kaggle_pcam_histopathology"]["seed"], "nomad": PACKAGES["kaggle_nomad2018_conductors"]["seed"],
         "openvaccine": PACKAGES["kaggle_openvaccine_degradation"]["seed"], "patent": PACKAGES["kaggle_patent_phrase_matching"]["seed"]}


@needs_packages
class TaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kaggle-tasks-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patcher = mock.patch.object(kaggle_data, "data_dir", lambda: self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_source(self, task, source, budget=30):
        path = self.tmp / f"{task}_{abs(hash(source))}.py"
        path.write_text(source)
        module = sandboxed.source_program(path, name=f"kaggle_{task}_{abs(hash(source))}")  # as the wrappers hand it over
        return getattr(kaggle_tasks, task)(module, {"budget_s": budget})

    def test_seeds_run_on_synthetic_data(self):
        for task, make in FAKES.items():
            with self.subTest(task=task):
                make(self.tmp)
                value, details = self.run_source(task, SEEDS[task])
                self.assertTrue(math.isfinite(value))
                self.assertIn("fit_predict_seconds", details)

    def test_jigsaw_metric_is_macro_auc(self):
        from sklearn.metrics import roc_auc_score
        truth = fake_jigsaw(self.tmp)
        noise = RNG.rand(*truth.shape)
        value, details = self.run_source("jigsaw", "import numpy as np\nS = np.array(%s)\n\ndef fit_predict(a, b, c):\n    return S\n" % noise.tolist())
        expected = np.mean([roc_auc_score(truth[:, j], noise[:, j]) for j in range(6)])
        self.assertAlmostEqual(value, expected, places=12)
        self.assertEqual(set(details["per_class_auc"]), set(CLASSES))

    def test_noisyoffice_metric_pools_pixels(self):
        clean = fake_noisyoffice(self.tmp)
        value, _ = self.run_source("noisyoffice", "def fit_predict(train, test, budget):\n    return [n / 255.0 for n in test['noisy']]\n")
        with np.load(self.tmp / "noisyoffice/public/test_noisy.npz") as stored:
            noisy = [stored[f"a{i}"] for i in range(len(stored.files))]
        errors = np.concatenate([((n / 255.0) - (c / 255.0)).ravel() for n, c in zip(noisy, clean)])
        self.assertAlmostEqual(value, float(np.sqrt(np.mean(errors ** 2))), places=12)
        all_white, _ = self.run_source("noisyoffice", "def fit_predict(train, test, budget):\n    return [n * 0 + 1.0 for n in test['noisy']]\n")
        self.assertGreater(all_white, value)
        with self.assertRaises(runtime.CandidateError):
            self.run_source("noisyoffice", "def fit_predict(train, test, budget):\n    return [n[:, :-1] / 255.0 for n in test['noisy']]\n")

    def test_pcam_requires_probabilities(self):
        fake_pcam(self.tmp)
        with self.assertRaises(runtime.CandidateError) as caught:
            self.run_source("pcam", "def fit_predict(train, test, budget):\n    return test['images'].mean(axis=(1, 2, 3))\n")
        self.assertIn("[0, 1]", str(caught.exception))
        value, _ = self.run_source("pcam", "def fit_predict(train, test, budget):\n    return 1 - test['images'][:, 16:32, 16:32].mean(axis=(1, 2, 3)) / 255\n")
        self.assertGreater(value, 0.9)

    def test_nomad_metric_is_mean_rmsle(self):
        truth = fake_nomad(self.tmp)
        value, details = self.run_source("nomad", "import numpy as np\nT = np.array(%s)\n\ndef fit_predict(a, b, c):\n    return T * 1.1\n" % truth.tolist())
        expected = np.mean([np.sqrt(np.mean((np.log1p(truth[:, j] * 1.1) - np.log1p(truth[:, j])) ** 2)) for j in range(2)])
        self.assertAlmostEqual(value, expected, places=12)
        with self.assertRaises(runtime.CandidateError):
            self.run_source("nomad", "import numpy as np\n\ndef fit_predict(a, b, c):\n    return -2 * np.ones((len(b), 2))\n")

    def test_openvaccine_metric_is_mcrmse(self):
        truth = fake_openvaccine(self.tmp)
        value, details = self.run_source("openvaccine", "import numpy as np\n\ndef fit_predict(a, b, c):\n    return np.full((len(b), 68, 3), 0.5)\n")
        expected = np.mean([np.sqrt(np.mean((truth[:, :, j] - 0.5) ** 2)) for j in range(3)])
        self.assertAlmostEqual(value, expected, places=12)
        with self.assertRaises(runtime.CandidateError):
            self.run_source("openvaccine", "import numpy as np\n\ndef fit_predict(a, b, c):\n    return np.zeros((len(b), 107, 5))\n")

    def test_patent_metric_is_pearson(self):
        truth = fake_patent(self.tmp)
        pred = truth + RNG.rand(len(truth))
        value, _ = self.run_source("patent", "import numpy as np\nP = np.array(%s)\n\ndef fit_predict(a, b, c):\n    return P\n" % pred.tolist())
        self.assertAlmostEqual(value, float(np.corrcoef(truth, pred)[0, 1]), places=12)


@needs_packages
class DeriveTest(unittest.TestCase):
    def test_npz_bytes_is_byte_stable_and_loadable(self):
        arrays = {"a": np.arange(6).reshape(2, 3), "b": np.array(["x", "yz"])}
        first, second = kaggle_derive.npz_bytes(arrays), kaggle_derive.npz_bytes(arrays)
        self.assertEqual(first, second)
        with np.load(io.BytesIO(first), allow_pickle=False) as data:
            np.testing.assert_array_equal(data["a"], arrays["a"])

    def test_nomad_record_reproduces_cell_lengths_and_angles(self):
        a = 6.2
        hexagonal = np.array([[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, 23.6]])
        record = kaggle_derive.nomad_record(hexagonal, np.zeros((5, 3)), [13, 31, 8, 8, 8], 194)
        self.assertAlmostEqual(record["lattice_angle_gamma_degree"], 120.0, places=6)
        self.assertAlmostEqual(record["lattice_angle_alpha_degree"], 90.0, places=6)
        self.assertAlmostEqual(record["lattice_vector_2_ang"], a, places=6)
        self.assertEqual((record["percent_atom_al"], record["percent_atom_ga"], record["percent_atom_in"]), (0.5, 0.5, 0.0))

    def test_openvaccine_replays_the_mle_bench_sample(self):
        """MLE-bench samples 240 rows among SN_filter > 0 rows of the 2400-row file; the public file holds exactly those rows."""
        rng = np.random.RandomState(3)
        sn = rng.rand(2400) < 0.66
        full = pd.DataFrame({"SN_filter": sn.astype(int)})
        expected = set(full[full["SN_filter"] > 0].sample(n=240, random_state=0).index)
        rows = []
        for i in np.flatnonzero(sn):
            lists = {c: json.dumps([0.1] * 68) for c in kaggle_derive.OV_LIST_COLUMNS}
            rows.append({"index": int(i), "id": f"id_{i}", "sequence": "A" * 107, "structure": "." * 107, "predicted_loop_type": "E" * 107,
                         "signal_to_noise": 2.0, "SN_filter": 1, "seq_length": 107, "seq_scored": 68, **lists})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "train.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            files, summary = kaggle_derive.openvaccine({"openvaccine-trainset": path})
        self.assertEqual(set(summary["test_original_index"]), expected)
        self.assertEqual(summary["train"] + summary["test"], int(sn.sum()))

    def test_noisyoffice_split_is_by_text_passage(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow unavailable")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            fonts = [f"{s}{t}{e}" for s in "fnL" for t in "tsr" for e in "em"]
            for font in fonts:
                for part in ("TR", "VA", "TE"):
                    clean = np.full((4, 5), 200 + len(part), np.uint8)
                    png = io.BytesIO()
                    Image.fromarray(clean, "L").save(png, format="PNG")
                    archive.writestr(f"NoisyOffice/SimulatedNoisyOffice/clean_images_grayscale/Font{font}_Clean_{part}.png", png.getvalue())
                    for noise in "fwcp":
                        rgb = np.repeat(clean[..., None] - 50, 3, axis=2)
                        png = io.BytesIO()
                        Image.fromarray(rgb, "RGB").save(png, format="PNG")
                        archive.writestr(f"NoisyOffice/SimulatedNoisyOffice/simulated_noisy_images_grayscale/Font{font}_Noise{noise}_{part}.png", png.getvalue())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "noisyoffice.zip"
            path.write_bytes(buffer.getvalue())
            files, summary = kaggle_derive_images.noisyoffice({"noisyoffice": path})
        self.assertEqual((summary["train_images"], summary["test_images"]), (144, 72))
        self.assertTrue(all(name.endswith("_TE") for name in summary["test_order"]))
        with np.load(io.BytesIO(files["noisyoffice/private/test_clean.npz"])) as data:
            self.assertEqual(len(data.files), 72)


class FamilyTest(unittest.TestCase):
    def test_families_register_with_labels_and_upgrades(self):
        registered = PACKAGES
        self.assertEqual(set(registered), set(LABELS))
        for fields in LABELS.values():  # the five labels the generator's table holds for each family
            self.assertTrue(all(fields.get(key) for key in ("scientific_contribution", "artifact", "domain", "data", "verification")))
        upgrades = json.loads((ROOT / "catalog" / "upgrades" / "kaggle.json").read_text())
        for task_id, spec in upgrades["tasks"].items():
            self.assertIn(spec["family"], registered, task_id)
            self.assertTrue(spec["notes"], task_id)
            for item in spec["requires"]:
                if item.startswith("file:"):
                    self.assertTrue(item.startswith("file:data/kaggle/"), item)
            budget = spec["params"]["budget_s"]
            self.assertIn(f"time_budget_s = {budget}", registered[spec["family"]]["contract"], task_id)
            # the evaluator deadline must outlast loading, the budget, the kill grace and scoring
            limit = budget + kaggle_sandbox.LOAD_LIMIT_S + kaggle_sandbox.GRACE_S + 30
            self.assertGreaterEqual(registered[spec["family"]]["timeout"], limit, task_id)
        for name, spec in registered.items():
            compile(spec["seed"], name, "exec")


if __name__ == "__main__":
    unittest.main()
