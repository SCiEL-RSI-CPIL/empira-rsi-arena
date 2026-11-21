"""Reaction-yield prediction on two high-throughput experimentation datasets with their published splits.

Buchwald-Hartwig (Ahneman et al. 2018): the four out-of-sample tests
(sheets Test1-Test4 of Dreher_and_Doyle_input_data.xlsx), whose test reactions
use isoxazole additives that never occur in training; the training rows are the
first 3057/3055/3058/3055 rows as in rxn_yields. Reactions are given by their
components and by the reaction SMILES that rxn_yields builds (aryl halide,
p-toluidine, the Pd precatalyst, ligand, base and additive; product from the
coupling template). Suzuki-Miyaura (Perera et al. 2018): the ten random 70/30
splits of rxn_yields (first 4032 reactions of each split train).

Every split is predicted by ``predict_yields(train_reactions, train_yields,
test_reactions, info)`` in a fresh child process (chem_sandbox) that sees no
test yields; splits run separately because the test reactions of one split are
training reactions of another. Yields are percent; the score is the mean test
R^2 over splits (scikit-learn definition).
"""
from __future__ import annotations

import numpy as np

from . import chem_sandbox
from .chem_common import load_json, predictions, r2, regression_metrics

AMINE = "Cc1ccc(N)cc1"
CATALYST = "O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F"
TEMPLATE = ("[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>"
            "[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]")
PUBLISHED_BH = {"Test1": 0.84, "Test2": 0.84, "Test3": 0.75, "Test4": 0.49}  # Yield-BERT, Schwaller et al. 2021


def buchwald_hartwig_reactions(records):
    """Component dictionaries plus the rxn_yields reaction SMILES (RDKit canonical) for the evaluator's records."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdChemReactions
    RDLogger.DisableLog("rdApp.*")
    template = rdChemReactions.ReactionFromSmarts(TEMPLATE)
    amine = Chem.MolFromSmiles(AMINE)
    catalyst = Chem.MolToSmiles(Chem.MolFromSmiles(CATALYST))
    canonical, products, reactions = {}, {}, []
    for record in records:
        parts = {}
        for key in ("aryl_halide", "ligand", "base", "additive"):
            smiles = record[key]
            if smiles not in canonical:
                canonical[smiles] = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            parts[key] = canonical[smiles]
        halide = record["aryl_halide"]
        if halide not in products:
            outcomes = {Chem.MolToSmiles(p[0]) for p in template.RunReactants((Chem.MolFromSmiles(halide), amine))}
            if len(outcomes) != 1:
                raise RuntimeError(f"coupling template is ambiguous for {halide}")
            products[halide] = outcomes.pop()
        reactants = ".".join([parts["aryl_halide"], AMINE, catalyst, parts["ligand"], parts["base"], parts["additive"]])
        reactions.append({**parts, "amine": AMINE, "catalyst": catalyst, "product": products[halide],
                          "reaction_smiles": f"{reactants}>>{products[halide]}"})
    return reactions


def _split_score(module, name, train_reactions, train_yields, test_reactions, test_yields, wall_seconds, dataset):
    info = {"dataset": dataset, "split": name, "n_train": len(train_reactions), "n_test": len(test_reactions), "yield_unit": "percent"}
    raw = chem_sandbox.run_candidate(module, "predict_yields", [train_reactions, list(map(float, train_yields)), test_reactions, info],
                                     wall_seconds=wall_seconds)
    predicted = predictions(raw, len(test_reactions))
    return {"split": name, **regression_metrics(test_yields, predicted)}


def buchwald_hartwig_score(module, wall_seconds_per_split=200):
    """Mean R^2 over Ahneman's four out-of-sample additive tests; higher is better."""
    data = load_json("yields", "buchwald_hartwig.json")
    rows = []
    for name, split in data["tests"].items():
        train = buchwald_hartwig_reactions(split["train"])
        test = buchwald_hartwig_reactions(split["test"])
        rows.append(_split_score(module, name, train, [r["yield"] for r in split["train"]], test,
                                 np.array([r["yield"] for r in split["test"]]), wall_seconds_per_split, "buchwald_hartwig"))
        rows[-1]["published_yield_bert_r2"] = PUBLISHED_BH[name]
    return float(np.mean([r["r2"] for r in rows])), {"splits": rows, "mean_r2": float(np.mean([r["r2"] for r in rows]))}


def suzuki_miyaura_score(module, wall_seconds_per_split=80):
    """Mean R^2 over the ten random 70/30 splits; higher is better."""
    data = load_json("yields", "suzuki_miyaura.json")
    yields = 100.0 * np.asarray(data["yields_fraction"], dtype=float)
    reactions = [{"reaction_smiles": r} for r in data["reactions"]]
    rows = []
    for k, order in enumerate(data["split_orders"]):
        train, test = order[:data["train_size"]], order[data["train_size"]:]
        rows.append(_split_score(module, f"random_split_{k}", [reactions[i] for i in train], yields[train],
                                 [reactions[i] for i in test], yields[test], wall_seconds_per_split, "suzuki_miyaura"))
    scores = [r["r2"] for r in rows]
    return float(np.mean(scores)), {"splits": rows, "mean_r2": float(np.mean(scores)), "std_r2": float(np.std(scores)),
                                    "min_r2": float(min(scores))}


__all__ = ["buchwald_hartwig_score", "suzuki_miyaura_score", "buchwald_hartwig_reactions", "r2"]
