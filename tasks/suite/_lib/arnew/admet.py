"""TDC ADMET benchmark group tasks: train on the official split and score on the official test set (5 seeds).

Requires PyTDC and rdkit. Contract: fit_predict(train_smiles, train_y, valid_smiles, valid_y, test_smiles, info) returns
one prediction per test molecule (probability or value). The test labels are never given to the candidate: every call
runs in a fresh sandboxed child process (the old suite's chem_sandbox, as for Matbench) that receives only these arguments,
cannot read data/ or this library, and cannot start processes or open sockets. Until 2026-11-15 the candidate was called
inside the evaluator process, where it could take the labels from the caller's frame.
"""
from __future__ import annotations

import numpy as np

from .common import CandidateError, candidate_sandbox, data_dir, require

SEEDS = (1, 2, 3, 4, 5)
WALL_SECONDS_PER_SEED = 700  # five calls inside the task's 3,600 s; the slowest starting program needs about 300 s for one
MEMORY_MB = 6144


def _reexec_in_venv():
    """The harness interpreter has no rdkit or PyTDC: restart this evaluator under tools/venv-admet, once."""
    import os
    import sys
    from pathlib import Path

    python = Path(__file__).resolve().parents[3] / "tools" / "venv-admet" / "bin" / "python"
    if python.is_file() and not os.environ.get("ARNEW_ADMET_REEXEC"):
        os.environ["ARNEW_ADMET_REEXEC"] = "1"
        os.execv(str(python), [str(python)] + sys.argv)


def run(module, name, metric):
    try:
        import tdc  # noqa: F401
        import rdkit  # noqa: F401
    except ImportError:
        _reexec_in_venv()
        raise
    from tdc.benchmark_group import admet_group

    group = admet_group(path=str(data_dir("admet")))
    sandbox = candidate_sandbox()
    predictions_list = []
    for seed in SEEDS:
        benchmark = group.get(name)
        train, valid = group.get_train_valid_split(benchmark=benchmark["name"], split_type="default", seed=seed)
        test = benchmark["test"]
        info = {"name": benchmark["name"], "metric": metric, "kind": "regression" if metric in ("mae", "spearman") else "binary", "seed": seed}
        raw = sandbox.run_candidate(module, "fit_predict",
                                    [train["Drug"].tolist(), train["Y"].tolist(), valid["Drug"].tolist(), valid["Y"].tolist(), test["Drug"].tolist(), info],
                                    wall_seconds=WALL_SECONDS_PER_SEED, memory_mb=MEMORY_MB, seed=seed)
        try:
            pred = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise CandidateError("return one number per test molecule") from None
        require(len(pred) == len(test), "return one prediction per test molecule")
        require(bool(np.isfinite(pred).all()), "predictions must be finite")
        predictions_list.append({benchmark["name"]: pred})
    result = group.evaluate_many(predictions_list)
    mean, std = result[benchmark["name"]]
    return float(mean), {"metric": metric, "mean": float(mean), "std": float(std), "seeds": len(SEEDS), "verification": "official_tdc_evaluate_many"}
