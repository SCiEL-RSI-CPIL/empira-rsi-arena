"""Alignment accuracy task: SP/TC definitions, self-consistency and contract checks."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import bio_align as al  # noqa: E402
from autoresearch_suite.bio_physics_common import data_dir  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

PREPARED = (data_dir() / al.DATA).is_file()

SAMPLE_XML = """<macsim><alignment><aln-name>test</aln-name>
<sequence seq-type="Protein"><seq-name>a</seq-name><seq-info></seq-info>
<seq-data>ACD-EF
</seq-data></sequence>
<sequence seq-type="Protein"><seq-name>b</seq-name><seq-info></seq-info>
<seq-data>ACDQEF
</seq-data></sequence>
<column-score><colsco-name>coreblock</colsco-name><colsco-data>
1 1 1 0 1 1 </colsco-data></column-score></alignment></macsim>"""


class DefinitionTests(unittest.TestCase):
    def test_reference_parsing(self):
        names, rows, mask = al.parse_reference(SAMPLE_XML)
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(rows, ["ACD-EF", "ACDQEF"])
        self.assertEqual(mask, [1, 1, 1, 0, 1, 1])

    def test_reference_alignment_scores_one(self):
        names, rows, mask = al.parse_reference(SAMPLE_XML)
        case = {"names": names, "reference": rows, "core": mask, "sequences": [r.replace("-", "") for r in rows]}
        sp, tc = al.case_scores(case, rows)
        self.assertAlmostEqual(sp, 1.0, places=12)
        self.assertAlmostEqual(tc, 1.0, places=12)

    def test_wrong_alignment_scores_zero(self):
        names, rows, mask = al.parse_reference(SAMPLE_XML)
        case = {"names": names, "reference": rows, "core": mask, "sequences": [r.replace("-", "") for r in rows]}
        # No residue of one sequence shares a column with any residue of the other.
        sp, tc = al.case_scores(case, ["-----ACDEF", "ACDQEF----"])
        self.assertAlmostEqual(sp, 0.0, places=12)
        self.assertAlmostEqual(tc, 0.0, places=12)

    def test_pair_counting_on_a_hand_checked_case(self):
        # Three sequences, two core columns: each core column has three residues -> three pairs each.
        case = {"reference": ["AB", "AB", "AB"], "core": [1, 1], "sequences": ["AB", "AB", "AB"]}
        pairs, columns = al._core_pairs(case["reference"], case["core"])
        self.assertEqual(len(pairs), 6)
        self.assertEqual(len(columns), 2)
        sp, tc = al.case_scores(case, ["A-B", "AB-", "A-B"])
        self.assertAlmostEqual(sp, 4 / 6, places=12)  # the A column stays, B pairs only between rows 0 and 2
        self.assertAlmostEqual(tc, 0.5, places=12)


class ContractTests(unittest.TestCase):
    def test_valid_alignment_accepted(self):
        rows = al.check_alignment(["AC-D", "ACQD"], ["ACD", "ACQD"])
        self.assertEqual(rows, ["AC-D", "ACQD"])

    def test_invalid_alignments_rejected(self):
        for alignment, sequences in ((["ACD"], ["ACD", "ACQD"]), (["ACD", "ACQ"], ["ACD", "ACQD"]),
                                     (["AC-D", "ACQ"], ["ACD", "ACQD"]), ([1, 2], ["ACD", "ACQD"]),
                                     (["A" + "-" * 100, "ACQD" + "-" * 97], ["A", "ACQD"])):
            with self.assertRaises(CandidateError):
                al.check_alignment(alignment, sequences)


@unittest.skipUnless(PREPARED, "prepared data missing; run scripts/prepare_bio_physics.py --only bio_align")
class PreparedDataTests(unittest.TestCase):
    def test_all_instances_present_and_consistent(self):
        cases = al.load_cases()
        self.assertEqual(len(cases), 26)
        for name, case in cases:
            lengths = {len(row) for row in case["reference"]}
            self.assertEqual(len(lengths), 1, name)
            self.assertEqual(len(case["core"]), lengths.pop(), name)
            self.assertEqual([row.replace("-", "") for row in case["reference"]], case["sequences"], name)
            self.assertGreaterEqual(sum(case["core"]), 20, name)
            self.assertLessEqual(len(case["sequences"]), 10, name)

    def test_reference_alignments_score_one_everywhere(self):
        for name, case in al.load_cases():
            sp, tc = al.case_scores(case, case["reference"])
            self.assertAlmostEqual(sp, 1.0, places=12, msg=name)
            self.assertAlmostEqual(tc, 1.0, places=12, msg=name)

    def test_gapless_alignment_is_valid_but_poor(self):
        name, case = al.load_cases()[0]
        width = max(len(s) for s in case["sequences"])
        padded = [s + "-" * (width - len(s)) for s in case["sequences"]]
        sp, tc = al.case_scores(case, padded)
        self.assertGreaterEqual(sp, 0.0)
        self.assertLess(sp, 0.9)


if __name__ == "__main__":
    unittest.main()
