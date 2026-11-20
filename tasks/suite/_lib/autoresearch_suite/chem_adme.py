"""ADME property prediction on the Biogen public set (Fang et al. 2023) with the authors' official splits.

The candidate's ``predict(train_smiles, train_values, test_smiles, info)`` runs
in a child process (chem_sandbox) that receives the training molecules with
their measured values and the test molecules without values; the test values
stay in the evaluator. Values are log10 of the measured property as published
in molecularinformatics/Computational-ADME (MPNN/ADME_<endpoint>_{train,test}.csv).
"""
from __future__ import annotations

import csv

import numpy as np

from . import chem_sandbox
from .chem_common import data_file, mae, predictions, regression_metrics

ENDPOINTS = {
    "solubility": "log10 aqueous solubility at pH 6.8 (ug/mL)",
    "hlm": "log10 human liver microsomal intrinsic clearance CLint (mL/min/kg)",
    "mdr1_er": "log10 MDR1-MDCK efflux ratio (B-A/A-B)",
    "rlm": "log10 rat liver microsomal intrinsic clearance CLint (mL/min/kg)",
}


def load(endpoint):
    """(train_smiles, train_values, test_smiles, test_values) of one endpoint."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown ADME endpoint {endpoint}")
    parts = []
    for part in ("train", "test"):
        with data_file("adme", f"{endpoint}_{part}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        parts += [[r["smiles"] for r in rows], np.array([float(r["activity"]) for r in rows])]
    return tuple(parts)


def adme_score(module, endpoint, wall_seconds=540):
    """Mean absolute error (log10 units) of the candidate's test-set predictions; lower is better."""
    train_smiles, train_values, test_smiles, test_values = load(endpoint)
    info = {"endpoint": endpoint, "property": ENDPOINTS[endpoint], "n_train": len(train_smiles), "n_test": len(test_smiles)}
    raw = chem_sandbox.run_candidate(module, "predict", [train_smiles, train_values.tolist(), test_smiles, info],
                                     wall_seconds=wall_seconds)
    predicted = predictions(raw, len(test_smiles))
    metrics = regression_metrics(test_values, predicted)
    baseline = mae(test_values, np.full(len(test_values), float(np.mean(train_values))))
    return metrics["mae"], {**metrics, "endpoint": endpoint, "property": ENDPOINTS[endpoint], "n_train": len(train_smiles),
                            "train_mean_predictor_mae": baseline}
