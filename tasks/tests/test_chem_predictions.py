"""Prediction tasks (ADME, Matbench, reaction yields): metrics, parsers, label hiding and prepared splits."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import chem_common, chem_sandbox, chem_sources  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import needs  # noqa: E402

LOCK = json.loads((ROOT / "catalog" / "chemistry-data-lock.json").read_text())
CANDIDATE_FILES = []


def tearDownModule():
    for path in CANDIDATE_FILES:
        Path(path).unlink(missing_ok=True)


def prepared(*paths):
    return all((chem_common.data_dir() / p).is_file() for p in paths)


def candidate(source):
    """A candidate module object backed by a real file, as the evaluator template provides it."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    handle.write(source)
    handle.close()
    CANDIDATE_FILES.append(handle.name)
    module = types.ModuleType("candidate_program")
    module.__file__ = handle.name
    exec(compile(source, handle.name, "exec"), module.__dict__)
    return module


class MetricTests(unittest.TestCase):
    def test_regression_metrics(self):
        y, p = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 2.0, 2.0, 5.0])
        self.assertAlmostEqual(chem_common.mae(y, p), 0.625)
        self.assertAlmostEqual(chem_common.r2(y, p), 1 - (0.25 + 0 + 1 + 1) / 5.0)
        self.assertAlmostEqual(chem_common.spearman([1, 2, 3, 4], [10, 20, 20, 40]), np.corrcoef([1, 2, 3, 4], [1, 2.5, 2.5, 4])[0, 1])

    def test_metrics_agree_with_scikit_learn_and_scipy(self):
        try:
            from scipy.stats import spearmanr
            from sklearn.metrics import mean_absolute_error, r2_score
        except ImportError:
            self.skipTest("scikit-learn and scipy are optional")
        rng = np.random.default_rng(1)
        y, p = rng.normal(size=200), rng.normal(size=200)
        p[:20] = p[0]  # ties
        self.assertAlmostEqual(chem_common.r2(y, p), r2_score(y, p), places=12)
        self.assertAlmostEqual(chem_common.mae(y, p), mean_absolute_error(y, p), places=12)
        self.assertAlmostEqual(chem_common.spearman(y, p), spearmanr(y, p)[0], places=12)

    def test_predictions_are_validated(self):
        np.testing.assert_allclose(chem_common.predictions([1, 2.5], 2), [1.0, 2.5])
        for bad in ([1.0], [1.0, float("nan")], {"a": 1}, "12", [[1.0, 2.0]], [None, 1.0]):
            with self.assertRaises(CandidateError):
                chem_common.predictions(bad, 2)


class ParserTests(unittest.TestCase):
    def test_formulas(self):
        self.assertEqual(chem_sources.parse_formula("Ag(AuS)2"), {"Ag": 1.0, "Au": 2.0, "S": 2.0})
        self.assertEqual(chem_sources.parse_formula("Ag0.5Ge1Pb1.75S4"), {"Ag": 0.5, "Ge": 1.0, "Pb": 1.75, "S": 4.0})
        self.assertEqual(chem_sources.parse_formula("Ca3(Al(SiO4)2)2"), {"Ca": 3.0, "Al": 2.0, "Si": 4.0, "O": 16.0})
        for bad in ("Ag(Au", "agS", "Fe-2"):
            with self.assertRaises(ValueError):
                chem_sources.parse_formula(bad)

    def test_xlsx_reader_resolves_shared_and_inline_strings(self):
        sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
                 '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1" t="inlineStr"><is><t>z</t></is></c></row>'
                 '<row r="2"><c r="B2"><v>3.5</v></c></row></sheetData></worksheet>')
        files = {
            "xl/workbook.xml": '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="S" sheetId="1" r:id="rId1"/></sheets></workbook>',
            "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
            "xl/sharedStrings.xml": '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>head</t></si></sst>',
            "xl/worksheets/sheet1.xml": sheet}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, text in files.items():
                archive.writestr(name, text)
        self.assertEqual(chem_sources.read_xlsx(buffer.getvalue()), {"S": [["head", None, "z"], [None, "3.5"]]})


class SandboxTests(unittest.TestCase):
    def test_candidate_sees_its_arguments_but_not_prepared_data_or_processes(self):
        # The probe stands for prepared data. It is written below a temporary task cache, not below the real one.
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        redirect = patch.dict(os.environ, {"ARENA_TASK_CACHE_DIR": cache.name})
        redirect.start()
        self.addCleanup(redirect.stop)
        target = chem_common.data_dir() / "probe.txt"
        source = f'''
import os
def run(values):
    out = {{"sum": sum(values)}}
    for name, action in (("data", lambda: open({str(target)!r}).read()),
                         ("library", lambda: open({str(Path(chem_common.__file__))!r}).read()),
                         ("process", lambda: os.system("true")),
                         ("network", lambda: __import__("socket").create_connection(("example.org", 80), timeout=1))):
        try:
            action()
            out[name] = "allowed"
        except PermissionError:
            out[name] = "denied"
    return out
'''
        target.parent.mkdir(parents=True, exist_ok=True)
        created = not target.exists()
        if created:
            target.write_text("secret")
        try:
            result = chem_sandbox.run_candidate(candidate(source), "run", [[1, 2, 3]], wall_seconds=60)
        finally:
            if created:
                target.unlink()
        self.assertEqual(result, {"sum": 6, "data": "denied", "library": "denied", "process": "denied", "network": "denied"})

    def test_failures_are_candidate_errors(self):
        module = candidate("def run(x):\n    raise ValueError('boom')\n")
        with self.assertRaises(CandidateError):
            chem_sandbox.run_candidate(module, "run", [1], wall_seconds=60)
        with self.assertRaises(CandidateError):
            chem_sandbox.run_candidate(candidate("import time\ndef run(x):\n    time.sleep(30)\n"), "run", [1], wall_seconds=3)
        with self.assertRaises(CandidateError):
            chem_sandbox.run_candidate(module, "missing", [1], wall_seconds=60)


@unittest.skipUnless(prepared(*[d["path"] for d in LOCK["derived"]]), "run python3 scripts/prepare_chemistry.py first")
class PreparedDataTests(unittest.TestCase):
    def test_prepared_files_match_the_lock(self):
        for entry in LOCK["derived"]:
            data = (chem_common.data_dir() / entry["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"], entry["path"])

    def test_matbench_folds_partition_each_dataset(self):
        from autoresearch_suite import chem_matbench
        sizes = {"steels": 312, "expt_gap": 4604, "jdft2d": 636, "phonons": 1265, "dielectric": 4764, "log_gvrh": 10987,
                 "perovskites": 18928, "expt_is_metal": 4921}
        for task, n in sizes.items():
            data = chem_matbench.load(task)
            self.assertEqual(len(data["targets"]), n)
            self.assertEqual(sorted(i for fold in data["test_folds"] for i in fold), list(range(n)))

    def test_buchwald_hartwig_tests_hold_out_additives(self):
        data = chem_common.load_json("yields", "buchwald_hartwig.json")
        sizes = {"Test1": (3057, 898), "Test2": (3055, 900), "Test3": (3058, 897), "Test4": (3055, 900)}
        for name, split in data["tests"].items():
            self.assertEqual((len(split["train"]), len(split["test"])), sizes[name])
            self.assertFalse({r["additive"] for r in split["train"]} & {r["additive"] for r in split["test"]})

    @needs("rdkit")
    def test_buchwald_hartwig_reaction_smiles_follow_rxn_yields(self):
        from autoresearch_suite import chem_yields
        record = chem_common.load_json("yields", "buchwald_hartwig.json")["tests"]["Test1"]["test"][0]
        reaction = chem_yields.buchwald_hartwig_reactions([record])[0]
        reactants, product = reaction["reaction_smiles"].split(">>")
        self.assertEqual(reactants.split(".")[1], "Cc1ccc(N)cc1")
        self.assertIn("[Pd]", reaction["catalyst"])
        self.assertIn("N", product)

    def test_training_mean_candidate_scores_the_reported_baseline(self):
        from autoresearch_suite import chem_adme
        module = candidate("def predict(train_smiles, train_values, test_smiles, info):\n"
                           "    return [sum(train_values) / len(train_values)] * len(test_smiles)\n")
        score, details = chem_adme.adme_score(module, "solubility", wall_seconds=60)
        self.assertAlmostEqual(score, details["train_mean_predictor_mae"], places=12)
        self.assertEqual(details["n"], 435)

    def test_classification_accepts_labels_only_and_scores_balanced_accuracy(self):
        from autoresearch_suite import chem_matbench
        truth = np.array([1, 1, 0, 0, 0])
        metrics = chem_matbench.classification_metrics(truth, np.array([1, 0, 0, 0, 1]))
        self.assertAlmostEqual(metrics["rocauc"], metrics["balanced_accuracy"])
        self.assertAlmostEqual(metrics["rocauc"], (0.5 + 2 / 3) / 2)
        module = candidate("def fit_predict(train_inputs, train_targets, test_inputs, info):\n    return [0.7] * len(test_inputs)\n")
        with self.assertRaises(CandidateError):
            chem_matbench.matbench_score(module, "expt_is_metal", wall_seconds_per_fold=60)

    def test_wrong_length_predictions_are_rejected(self):
        from autoresearch_suite import chem_matbench
        module = candidate("def fit_predict(train_inputs, train_targets, test_inputs, info):\n    return [0.0] * (len(test_inputs) - 1)\n")
        with self.assertRaises(CandidateError):
            chem_matbench.matbench_score(module, "steels", wall_seconds_per_fold=60)


if __name__ == "__main__":
    unittest.main()
