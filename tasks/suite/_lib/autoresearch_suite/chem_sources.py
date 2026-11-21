"""Derivations of the prepared chemistry files from the pinned raw downloads (standard library only).

``scripts/prepare_chemistry.py`` downloads every raw source listed in
``catalog/chemistry-data-lock.json``, verifies its SHA-256 and calls the
deriver registered here for each prepared file; derived files are byte-stable
functions of the raw payloads and are verified against the lock as well.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET

SHEET_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
MATBENCH = {
    "steels": ("composition", "yield strength", "MPa"),
    "expt_gap": ("composition", "gap expt", "eV"),
    "jdft2d": ("structure", "exfoliation_en", "meV/atom"),
    "phonons": ("structure", "last phdos peak", "cm^-1"),
    "dielectric": ("structure", "n", "dimensionless"),
    "log_gvrh": ("structure", "log10(G_VRH)", "log10(GPa)"),
    "perovskites": ("structure", "e_form", "eV/unit cell"),
    "expt_is_metal": ("composition", "is_metal", "1 = metal, 0 = non-metal"),
}
CLASSIFICATION = {"expt_is_metal"}
BH_SPLITS = (("Test1", 3058), ("Test2", 3056), ("Test3", 3059), ("Test4", 3056))  # rxn_yields NAME_SPLIT (1-based)
SM_TRAIN = 4032


def dumps(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def read_xlsx(payload):
    """Rows of every sheet of an .xlsx file as lists of strings (shared and inline strings resolved)."""
    archive = zipfile.ZipFile(io.BytesIO(payload))
    shared = []
    if "xl/sharedStrings.xml" in archive.namelist():
        for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", SHEET_NS):
            shared.append("".join(t.text or "" for t in item.iter("{%s}t" % SHEET_NS["m"])))
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    targets = {r.get("Id"): r.get("Target") for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
    sheets = {}
    for sheet in workbook.find("m:sheets", SHEET_NS):
        target = targets[sheet.get("{%s}id" % SHEET_NS["r"])]
        target = target if target.startswith("xl/") else "xl/" + target.lstrip("/")
        rows = []
        for row in ET.fromstring(archive.read(target)).iter("{%s}row" % SHEET_NS["m"]):
            cells = {}
            for cell in row.findall("m:c", SHEET_NS):
                column = 0
                for ch in re.match(r"[A-Z]+", cell.get("r")).group(0):
                    column = column * 26 + ord(ch) - 64
                value = cell.find("m:v", SHEET_NS)
                if cell.get("t") == "inlineStr":
                    text = "".join(t.text or "" for t in cell.iter("{%s}t" % SHEET_NS["m"]))
                elif value is None:
                    continue
                else:
                    text = shared[int(value.text)] if cell.get("t") == "s" else value.text
                cells[column - 1] = text
            if cells:
                rows.append([cells.get(i) for i in range(max(cells) + 1)])
        sheets[sheet.get("name")] = rows
    return sheets


def parse_formula(formula):
    """Element amounts of a formula such as 'Ag(AuS)2' or 'Fe0.62C0.00095'; parentheses may nest."""
    tokens = re.findall(r"[A-Z][a-z]?|\d+(?:\.\d+)?|\.\d+|\(|\)", formula)
    if "".join(tokens) != formula.replace(" ", ""):
        raise ValueError(f"cannot parse formula {formula!r}")
    position = 0

    def number():
        nonlocal position
        if position < len(tokens) and re.fullmatch(r"\d+(?:\.\d+)?|\.\d+", tokens[position]):
            position += 1
            return float(tokens[position - 1])
        return 1.0

    def group():
        nonlocal position
        amounts = {}
        while position < len(tokens) and tokens[position] != ")":
            token = tokens[position]
            if token == "(":
                position += 1
                inner = group()
                if position >= len(tokens) or tokens[position] != ")":
                    raise ValueError(f"unbalanced parentheses in {formula!r}")
                position += 1
                factor = number()
                for element, amount in inner.items():
                    amounts[element] = amounts.get(element, 0.0) + amount * factor
            elif re.fullmatch(r"[A-Z][a-z]?", token):
                position += 1
                amounts[token] = amounts.get(token, 0.0) + number()
            else:
                raise ValueError(f"unexpected token {token!r} in {formula!r}")
        return amounts

    amounts = group()
    if position != len(tokens) or not amounts:
        raise ValueError(f"cannot parse formula {formula!r}")
    return {element: round(amount, 12) for element, amount in amounts.items()}


def structure_record(structure):
    """Plain lattice/species/fractional-coordinate record of an ordered pymatgen Structure dict."""
    species, frac = [], []
    for site in structure["sites"]:
        if len(site["species"]) != 1 or abs(site["species"][0]["occu"] - 1.0) > 1e-9:
            raise ValueError("disordered site")
        species.append(site["species"][0]["element"])
        frac.append([float(v) for v in site["abc"]])
    return {"lattice": [[float(v) for v in row] for row in structure["lattice"]["matrix"]], "species": species,
            "frac_coords": frac}


def matbench_deriver(task):
    input_type, target, unit = MATBENCH[task]

    def derive(payloads):
        raw = json.loads(gzip.decompress(payloads[f"matbench-{task.replace('_', '-')}"]))
        if raw["columns"] != [input_type, target]:
            raise ValueError(f"unexpected Matbench columns {raw['columns']}")
        n = len(raw["data"])
        digits = len(str(n))
        ids = [f"mb-{task.replace('_', '-')}-{i + 1:0{digits}d}" for i in raw["index"]]
        position = {identifier: k for k, identifier in enumerate(ids)}
        folds = json.loads(payloads["matbench-validation"])["splits"][f"matbench_{task}"]
        test_folds = [sorted(position[i] for i in folds[f"fold_{k}"]["test"]) for k in range(5)]
        if sorted(i for fold in test_folds for i in fold) != list(range(n)):
            raise ValueError("Matbench test folds do not partition the dataset")
        for k in range(5):
            train = sorted(position[i] for i in folds[f"fold_{k}"]["train"])
            if sorted(train + test_folds[k]) != list(range(n)):
                raise ValueError(f"fold {k} train and test do not partition the dataset")
        if input_type == "composition":
            inputs = [{"formula": row[0], "composition": parse_formula(row[0])} for row in raw["data"]]
        else:
            inputs = [structure_record(row[0]) for row in raw["data"]]
        return dumps({"task": f"matbench_{task}", "task_type": "classification" if task in CLASSIFICATION else "regression",
                      "input_type": input_type, "target": target, "unit": unit, "ids": ids,
                      "inputs": inputs, "targets": [float(row[1]) for row in raw["data"]], "test_folds": test_folds})
    return derive


def derive_buchwald_hartwig(payloads):
    sheets = read_xlsx(payloads["rxn-yields-buchwald-hartwig"])
    tests = {}
    for name, split in BH_SPLITS:
        rows = sheets[name]
        if rows[0][:5] != ["Ligand", "Additive", "Base", "Aryl halide", "Output"]:
            raise ValueError(f"unexpected header in sheet {name}")
        records = [{"ligand": r[0], "additive": r[1], "base": r[2], "aryl_halide": r[3], "yield": float(r[4])} for r in rows[1:]]
        train, test = records[:split - 1], records[split - 1:]
        if {r["additive"] for r in train} & {r["additive"] for r in test}:
            raise ValueError(f"{name}: additives overlap between training and test reactions")
        tests[name] = {"train": train, "test": test}
    return dumps({"source": "Dreher_and_Doyle_input_data.xlsx (Ahneman et al. 2018)", "tests": tests})


def derive_suzuki_miyaura(payloads):
    reactions, yields, orders = None, None, []
    for k in range(10):
        rows = list(csv.reader(io.StringIO(payloads[f"rxn-yields-suzuki-split-{k}"].decode("utf-8")), delimiter="\t"))
        if rows[0][1:] != ["rxn", "y"]:
            raise ValueError("unexpected Suzuki-Miyaura header")
        table = {int(r[0]): (r[1], float(r[2])) for r in rows[1:]}
        if sorted(table) != list(range(len(table))):
            raise ValueError("Suzuki-Miyaura split is not a permutation of the dataset")
        if reactions is None:
            reactions = [table[i][0] for i in range(len(table))]
            yields = [table[i][1] for i in range(len(table))]
        elif any(table[i] != (reactions[i], yields[i]) for i in range(len(table))):
            raise ValueError("Suzuki-Miyaura splits disagree on reactions or yields")
        orders.append([int(r[0]) for r in rows[1:]])
    return dumps({"source": "rxn_yields random_split_0..9 (Perera et al. 2018)", "train_size": SM_TRAIN,
                  "reactions": reactions, "yields_fraction": yields, "split_orders": orders})


DERIVERS = {
    "adme/solubility_train.csv": (("biogen-sol-train",), lambda p: p["biogen-sol-train"]),
    "adme/solubility_test.csv": (("biogen-sol-test",), lambda p: p["biogen-sol-test"]),
    "adme/hlm_train.csv": (("biogen-hlm-train",), lambda p: p["biogen-hlm-train"]),
    "adme/hlm_test.csv": (("biogen-hlm-test",), lambda p: p["biogen-hlm-test"]),
    "adme/mdr1_er_train.csv": (("biogen-mdr1-er-train",), lambda p: p["biogen-mdr1-er-train"]),
    "adme/mdr1_er_test.csv": (("biogen-mdr1-er-test",), lambda p: p["biogen-mdr1-er-test"]),
    "adme/rlm_train.csv": (("biogen-rlm-train",), lambda p: p["biogen-rlm-train"]),
    "adme/rlm_test.csv": (("biogen-rlm-test",), lambda p: p["biogen-rlm-test"]),
    "yields/buchwald_hartwig.json": (("rxn-yields-buchwald-hartwig",), derive_buchwald_hartwig),
    "yields/suzuki_miyaura.json": (tuple(f"rxn-yields-suzuki-split-{k}" for k in range(10)), derive_suzuki_miyaura),
    **{f"matbench/{task}.json": ((f"matbench-{task.replace('_', '-')}", "matbench-validation"), matbench_deriver(task))
       for task in MATBENCH},
}
