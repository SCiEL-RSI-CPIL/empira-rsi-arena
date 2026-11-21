"""Read the hash-pinned finance tables prepared by scripts/prepare_finance.py.

Evaluators never download anything. Each derived table is a small CSV written
by the preparation script from a pinned public source (see
catalog/finance-data-lock.json); its SHA-256 is checked here before use, so a
modified or partially written file is refused instead of silently scored.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from arena_task_paths import data_root

SETUP = "python3 scripts/prepare_finance.py"

# SHA-256 of every derived table; must equal the "derived" entries of
# catalog/finance-data-lock.json (checked by tests/test_finance_data.py).
DERIVED_SHA256 = {
    "ff_factors_monthly.csv": "2af86e4be533d6b244511b3587304987c08750043b41fd8aa19c21b17f94c390",
    "ind10_vw_monthly.csv": "858b46297d8e47513ec555fadf724dc43ea62550f1556adce86974b940f882ca",
    "ff25_vw_monthly.csv": "c8414c3aacc1c06619dba2bc07f3d72c854d6bd04accaa78036788884a41e80d",
    "ind49_vw_monthly.csv": "d4cce6fd50934f63e159a36520a387667ad7a7e05c934ed0e8d26215fe6ec18c",
    "goyal_welch_monthly.csv": "f15431437234fbfde36d592ceb34b9aaa2f1d4420377459cd1bcfe4008c7e4ff",
    "dax_2002_surface.csv": "c241abeef02a4f9429a26939d55612e94483b1e747e74ff17cfe77263271df68",
    "kidney_preflib_pools.csv": "3f6f248c99a42a9bcf8352fb6e21765e4ec8e01fe298677f7ae5a89a71062ad2",
    "ff_factors_daily.csv": "da1edec5aeb29085e637dfdaa7ccc0452d690a3baf3db814ad6dbca0fd83a38f",
    "ind49_vw_daily.csv": "08caaef561ae5a4be311e8a7bf24c79e0a7c8754ecc3ffec89ed4e79ced3638d",
    "gsw_yields_monthly.csv": "0d99c2597e2dcb5d2f3b836132421d41d669af0e49ceed356900b50d733743b8",
    "fredmd_2024_10.csv": "eebcaa9024b3412849a86445e843d853543fd1b0b767cc2c85a10875fbac56a2",
    "orlib_port1.csv": "1293606ba3b093704170717d7fa870df7ac6374edf511cbf99f1413a94fab31c",
    "orlib_portef1.csv": "13c25fac419d5e58f17f2b5fe9fa12e7642c921721ab1cfb89aa42405d634750",
    "orlib_port2.csv": "c9cc40c29e9ebdcbc3ff86a6457f730136c42f03a7424feb9820ca4cf29f78be",
    "orlib_portef2.csv": "c70450ea4923c7fe02e53dedda709a8198040b95820166a385724e169435a98c",
    "orlib_port3.csv": "244436c50e3a85e892dfaf3cef1648e36fc5a1d3bfad242c3736aa4941f91a87",
    "orlib_portef3.csv": "16c4bbe9a54c2c4b8ff99b06bf49b52cef5be08bf44d9737d4b79292d46868b7",
    "orlib_port4.csv": "f817d59d27a81e09f9dcf30d6f31eed7f1a34a8e17cb38b51d18858cc4f002a6",
    "orlib_portef4.csv": "71bcbbfe30a65360e9e7152bb2664a53430afd0dd884f014f172c00493d77b9e",
    "orlib_port5.csv": "550cb2bd438c077dea2d277973a445deb88a5970a8c29aacb881b6bbbe4b55f6",
    "orlib_portef5.csv": "a789145d0b0ae788e7f1b62301ba12a8175c88f7c2a9d6100e47e3f29c244384",
    "orlib_indtrack1.csv": "47ee06ba664c6e9f87b0a098be779aefb1345f40acba91d3d5045373721db618",
    "orlib_indtrack2.csv": "7d16bea33af3017ebc975b9531e295d351231fa57b497ad8ab979757048404db",
    "orlib_indtrack3.csv": "4d9e0b1c683231e1ebf080b4afd47288fbb1d2bdb79862c26f9d76f79b10836b",
    "orlib_indtrack4.csv": "63b44a7e27aaf89d6487ddf554f7e56aabda81ffdc76b77462a959ceb603648b",
    "orlib_indtrack5.csv": "d05dbfc489e3c9dc672d42a84c595b1a18c5cc9e34e1a95ef77787e9b7bea3b9",
}


def data_directory():
    return Path(os.environ.get("AUTORESEARCH_FINANCE_DATA", data_root() / "finance"))


def path(name):
    return data_directory() / name


@lru_cache(maxsize=None)
def read_bytes(name):
    file = path(name)
    if not file.is_file():
        raise RuntimeError(f"missing finance data file {name}; run `{SETUP}` (requires network once)")
    payload = file.read_bytes()
    expected = DERIVED_SHA256.get(name)
    if expected is None:
        raise RuntimeError(f"{name} is not a registered finance table")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise RuntimeError(f"{name} failed SHA-256 verification ({digest}); rerun `{SETUP}`")
    return payload


@lru_cache(maxsize=None)
def table(name):
    """Return (keys, columns, values): first CSV column as strings, the rest as float64."""
    rows = list(csv.reader(io.StringIO(read_bytes(name).decode("utf-8"))))
    header, body = rows[0], rows[1:]
    keys = tuple(row[0] for row in body)
    values = np.array([[float(x) if x not in ("", "NaN", "nan") else np.nan for x in row[1:]] for row in body], dtype=float)
    values.setflags(write=False)
    return keys, tuple(header[1:]), values


def window(name, start, end):
    """Rows of a table whose key lies in [start, end] (string comparison on fixed-width keys)."""
    keys, columns, values = table(name)
    index = [i for i, k in enumerate(keys) if start <= k <= end]
    if not index:
        raise RuntimeError(f"{name} has no rows between {start} and {end}")
    lo, hi = index[0], index[-1] + 1
    return keys[lo:hi], columns, values[lo:hi]
