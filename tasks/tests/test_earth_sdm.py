"""Species distribution models: the AUC, derivation of one region, label isolation and the seed."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_sdm as sdm  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
SDM_SEED = seed_source("science-earth-species-distribution-models")

LEAK_PROBE = '''
def predict_distributions(regions):
    import numpy as np
    keys = set()
    for region in regions.values():
        keys |= set(region)
        for group in region["sites"].values():
            keys |= set(group)
    assert "pa" not in keys and "labels" not in keys, sorted(keys)
    return {name: {s: np.zeros(len(g["x"])) for g in r["sites"].values() for s in g["species"]} for name, r in regions.items()}
'''


class AucTests(unittest.TestCase):
    def test_auc_counts_ties_as_half(self):
        self.assertAlmostEqual(ec.roc_auc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]), 1.0)
        self.assertAlmostEqual(ec.roc_auc([1, 0], [0.5, 0.5]), 0.5)
        self.assertAlmostEqual(ec.roc_auc([1, 1, 0, 0, 0], [3, 1, 2, 0, 1]), 4.5 / 6)


class DeriveTests(unittest.TestCase):
    def test_derive_reads_one_region_layout(self):
        header = "siteid,spid,x,y,occ,group,a,vegsys\n"
        files = {"train_po": header + "p1,s1,1,2,1,ba,0.5,3\np2,s2,1,2,1,ba,NA,1\n",
                 "train_bg": header + "b1,pseudo,3,4,0,NA,1.5,2\n"}
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name in sdm.RAW_SOURCES:
                key = name[len("nceas_"):]
                text = header + "x1,s1,0,0,1,ba,0,1\n"
                if "train_po" in key:
                    text = files["train_po"]
                elif "train_bg" in key:
                    text = files["train_bg"]
                elif "test_env" in key:
                    text = "group,siteid,x,y,a,vegsys\nba,t1,5,6,0.1,9\nba,t2,7,8,0.2,1\n"
                elif "test_pa" in key:
                    text = "group,siteid,x,y,s1,s2\nba,t1,5,6,1,0\nba,t2,7,8,0,1\n"
                path = Path(tmp) / name
                path.write_text(text)
                paths[name] = path
            nsw = sdm.derive(paths)[sdm.DATA]["regions"]["NSW"]
        self.assertEqual(nsw["covariates"], ["a", "vegsys"])
        self.assertEqual(nsw["categorical"], ["vegsys"])
        self.assertEqual(nsw["presence"]["s2"]["x"], [[None, 1.0]])
        self.assertEqual(nsw["groups"]["sr"]["pa"], [[1, 0], [0, 1]])


@unittest.skipUnless((ec.data_dir() / sdm.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_counts_and_label_isolation(self):
        public, labels = sdm.load()
        species = sum(len(g["species"]) for r in public.values() for g in r["sites"].values())
        self.assertEqual(species, 226)
        value, details = sdm.score(module_from(LEAK_PROBE))
        self.assertAlmostEqual(value, 0.5)  # constant scores: every AUC is one half

    def test_seed_and_invalid_output(self):
        value, details = sdm.score(module_from(SDM_SEED))
        self.assertGreater(value, 0.65)
        self.assertEqual(details["species"], 226)
        bad = SDM_SEED + "\n\ndef predict_distributions(regions):\n    return {}\n"
        with self.assertRaises(CandidateError):
            sdm.score(module_from(bad))


if __name__ == "__main__":
    unittest.main()
