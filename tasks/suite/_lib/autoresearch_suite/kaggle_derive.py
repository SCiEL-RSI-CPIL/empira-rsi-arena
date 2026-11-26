"""Derivation of the prepared files of the kaggle package from the pinned raw downloads.

Called by scripts/prepare_kaggle.py. Every function takes ``raw`` (source id -> local path of the verified
download) and returns ``{relative path: bytes}`` for files under ``data_dir()``: ``<task>/public/*`` (inputs
the candidate may read) and ``<task>/private/*`` (held-out labels). Writers are byte-stable (csv module,
compact JSON, zip entries with fixed timestamps) so the lock can record SHA-256 digests of derived files.
The image sources (NoisyOffice, PatchCamelyon) are in kaggle_derive_images.py.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile

import numpy as np
import pandas as pd

JIGSAW_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
OV_TRAIN_COLUMNS = ["index", "id", "sequence", "structure", "predicted_loop_type", "signal_to_noise", "SN_filter", "seq_length",
                    "seq_scored", "reactivity_error", "deg_error_Mg_pH10", "deg_error_pH10", "deg_error_Mg_50C", "deg_error_50C",
                    "reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
OV_TEST_COLUMNS = ["index", "id", "sequence", "structure", "predicted_loop_type", "seq_length", "seq_scored"]
OV_LIST_COLUMNS = OV_TRAIN_COLUMNS[9:]
OV_SCORED = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
NOMAD_TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
CATIONS = ("Al", "Ga", "In")
SYMBOL = {8: "O", 13: "Al", 31: "Ga", 49: "In"}


def csv_bytes(header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def jsonl_bytes(records):
    return "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in records).encode("utf-8")


def npz_bytes(arrays):
    """np.savez equivalent with fixed zip timestamps (byte-stable)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            with archive.open(info, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def seeded_test_mask(n, fraction, seed):
    """Boolean mask selecting round(fraction * n) rows by a seeded permutation (numpy RandomState, frozen stream)."""
    order = np.random.RandomState(seed).permutation(n)
    mask = np.zeros(n, dtype=bool)
    mask[order[:int(round(fraction * n))]] = True
    return mask


# --- Jigsaw ----------------------------------------------------------------------------------------------

def jigsaw(raw):
    read = dict(dtype={"id": str, "comment_text": str}, keep_default_na=False, na_filter=False)
    train = pd.read_csv(raw["jigsaw-train"], **read)
    test = pd.read_csv(raw["jigsaw-test"], **read)
    labels = pd.read_csv(raw["jigsaw-test-labels"], dtype={"id": str})
    assert list(train.columns) == ["id", "comment_text", *JIGSAW_CLASSES], list(train.columns)
    assert set(test["id"]) == set(labels["id"]) and len(test) == len(labels)
    assert not set(train["id"]) & set(test["id"])
    merged = test.merge(labels, on="id", how="inner", validate="one_to_one")
    scored = merged[merged[JIGSAW_CLASSES].sum(axis=1) >= 0]  # grade.py: rows marked -1 are not scored
    assert (scored[JIGSAW_CLASSES].isin([0, 1])).all().all()
    return {
        "jigsaw/public/train.csv": raw_bytes(raw["jigsaw-train"]),
        "jigsaw/public/test.csv": csv_bytes(["id", "comment_text"], scored[["id", "comment_text"]].itertuples(index=False)),
        "jigsaw/private/test_labels.csv": csv_bytes(["id", *JIGSAW_CLASSES],
                                                    scored[["id", *JIGSAW_CLASSES]].astype({c: int for c in JIGSAW_CLASSES}).itertuples(index=False)),
    }, {"train_rows": len(train), "test_rows": len(test), "scored_test_rows": len(scored)}


def raw_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


# --- NOMAD 2018 (ColabFit NMD-18) -------------------------------------------------------------------------

def _angle(u, v):
    cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def nomad_record(cell, positions, numbers, spacegroup):
    """Kaggle-style table columns for one structure (cation fractions, cell lengths and angles) plus the geometry."""
    cell = np.asarray(cell, dtype=float)
    elements = [SYMBOL[int(z)] for z in numbers]
    cations = [e for e in elements if e in CATIONS]
    a1, a2, a3 = cell
    record = {"spacegroup": int(spacegroup), "number_of_total_atoms": float(len(elements))}
    for symbol in CATIONS:
        record[f"percent_atom_{symbol.lower()}"] = round(cations.count(symbol) / len(cations), 6)
    for k, vector in enumerate(cell, start=1):
        record[f"lattice_vector_{k}_ang"] = round(float(np.linalg.norm(vector)), 6)
    record["lattice_angle_alpha_degree"] = round(_angle(a2, a3), 6)
    record["lattice_angle_beta_degree"] = round(_angle(a1, a3), 6)
    record["lattice_angle_gamma_degree"] = round(_angle(a1, a2), 6)
    record["lattice_vectors"] = [[round(float(v), 8) for v in row] for row in cell]
    record["elements"] = elements
    record["positions"] = [[round(float(v), 8) for v in row] for row in np.asarray(positions, dtype=float)]
    return record


def nomad(raw, test_fraction=0.1, seed=0):
    table = pd.read_parquet(raw["nmd18-colabfit"]).sort_values("configuration_id").reset_index(drop=True)
    assert len(table) == 3000 and table["configuration_id"].is_unique
    records = []
    for row in table.itertuples(index=False):
        (name,) = list(row.names)
        assert name.startswith("sg-")
        record = nomad_record(np.stack(row.cell), np.stack(row.positions), row.atomic_numbers, int(name[3:]))
        # ColabFit stores formation energy per atom; the competition's formation_energy_ev_natom is per cation
        # (x 5/2 for M2O3 cells; checked against the competition files, see validation/kaggle-checks.json).
        assert len(record["elements"]) == 5 * len([e for e in record["elements"] if e in CATIONS]) / 2
        record["formation_energy_ev_natom"] = round(float(row.formation_energy) * 2.5, 4)
        record["bandgap_energy_ev"] = round(float(row.electronic_band_gap), 4)
        records.append(record)
    mask = seeded_test_mask(len(records), test_fraction, seed)
    train = [r for r, m in zip(records, mask) if not m]
    test = [r for r, m in zip(records, mask) if m]
    for rows in (train, test):
        for k, r in enumerate(rows, start=1):
            r["id"] = k
    order = ["id", "spacegroup", "number_of_total_atoms", "percent_atom_al", "percent_atom_ga", "percent_atom_in",
             "lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang", "lattice_angle_alpha_degree",
             "lattice_angle_beta_degree", "lattice_angle_gamma_degree", "lattice_vectors", "elements", "positions"]
    return {
        "nomad/public/train.jsonl": jsonl_bytes([{k: r[k] for k in order + NOMAD_TARGETS} for r in train]),
        "nomad/public/test.jsonl": jsonl_bytes([{k: r[k] for k in order} for r in test]),
        "nomad/private/test_labels.csv": csv_bytes(["id", *NOMAD_TARGETS], [[r["id"], r[NOMAD_TARGETS[0]], r[NOMAD_TARGETS[1]]] for r in test]),
    }, {"materials": len(records), "train": len(train), "test": len(test)}


# --- OpenVaccine -------------------------------------------------------------------------------------------

def openvaccine(raw, original_rows=2400, test_size=0.1):
    """Replays MLE-bench's prepare.py: 240 test rows sampled (random_state=0) among the SN_filter > 0 rows.

    The public CSV holds exactly the SN_filter = 1 rows of the competition's train.json, in their original
    order and with their original row index, so ``sample`` draws the same rows MLE-bench draws.
    """
    frame = pd.read_csv(raw["openvaccine-trainset"], dtype={"signal_to_noise": str}).set_index("index", drop=False)
    assert frame.index.is_monotonic_increasing and frame.index.max() < original_rows and (frame["SN_filter"] > 0).all()
    # The only decimal column that is not a JSON list. pandas before 3.0 reads some of its 17-digit values one
    # unit in the last place off, and differently on arm64 and x86-64; float() is correctly rounded everywhere.
    frame["signal_to_noise"] = [float(text) for text in frame["signal_to_noise"]]
    for column in OV_LIST_COLUMNS:
        frame[column] = frame[column].map(json.loads)
    n_test = int(original_rows * test_size)
    test_index = frame[frame["SN_filter"] > 0].sample(n=n_test, random_state=0).index
    is_test = frame.index.isin(test_index)
    train, test = frame[~is_test].copy(), frame[is_test].copy()
    train["index"], test["index"] = range(len(train)), range(len(test))
    assert (test["seq_scored"] == 68).all() and (test["seq_length"] == 107).all()

    def records(table, columns):
        return [{c: (v.item() if hasattr(v, "item") else v) for c, v in zip(columns, row)} for row in table[columns].itertuples(index=False)]
    targets = np.stack([np.array([row[c] for c in OV_SCORED], dtype=np.float64).T for _, row in test.iterrows()])
    return {
        "openvaccine/public/train.jsonl": jsonl_bytes(records(train, OV_TRAIN_COLUMNS)),
        "openvaccine/public/test.jsonl": jsonl_bytes(records(test, OV_TEST_COLUMNS)),
        "openvaccine/private/test_labels.npz": npz_bytes({"targets": targets, "ids": np.array(test["id"].tolist(), dtype="U32")}),
    }, {"public_rows": len(frame), "train": len(train), "test": len(test), "test_original_index": sorted(int(i) for i in test_index)}


# --- U.S. patent phrase matching ---------------------------------------------------------------------------

def patent(raw):
    parts = {name: pd.read_parquet(raw[f"patent-{name}"]) for name in ("train", "validation", "test")}
    anchors = {name: set(part["anchor"]) for name, part in parts.items()}
    assert not (anchors["train"] & anchors["test"]) and not (anchors["validation"] & anchors["test"])
    train = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    test = parts["test"].reset_index(drop=True)
    train_rows = [[f"tr{k:05d}", r.anchor, r.target, r.context, float(r.score)] for k, r in enumerate(train.itertuples(index=False))]
    test_rows = [[f"te{k:05d}", r.anchor, r.target, r.context] for k, r in enumerate(test.itertuples(index=False))]
    return {
        "patent/public/train.csv": csv_bytes(["id", "anchor", "target", "context", "score"], train_rows),
        "patent/public/test.csv": csv_bytes(["id", "anchor", "target", "context"], test_rows),
        "patent/private/test_labels.csv": csv_bytes(["id", "score"], [[f"te{k:05d}", float(s)] for k, s in enumerate(test["score"])]),
    }, {"train": len(parts["train"]), "validation": len(parts["validation"]), "test": len(test),
        "anchors": {k: len(v) for k, v in anchors.items()}}
