"""Derivations of the small finance tables from pinned raw sources.

Used only by scripts/prepare_finance.py (never during scoring). Every function
maps {source_id: raw bytes} to CSV text. Numbers are copied as the original
text tokens, so a derived table is a byte-stable function of its source.
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile


def _zip_text(payload):
    archive = zipfile.ZipFile(io.BytesIO(payload))
    names = [n for n in archive.namelist() if not n.endswith("/")]
    if len(names) != 1:
        raise ValueError(f"expected one member in the archive, found {names}")
    return archive.read(names[0]).decode("latin-1")


def french_section(payload, title=None, key_width=6):
    """(header, rows) of one table in a Kenneth French CSV archive.

    ``title`` selects the table that follows a title line such as
    "Average Value Weighted Returns -- Monthly"; None selects the first table.
    Rows are kept while the first field is a ``key_width``-digit date.
    """
    lines = _zip_text(payload).splitlines()
    if title is None:
        start = next(i for i, line in enumerate(lines) if line.startswith(","))
    else:
        start = next(i for i, line in enumerate(lines) if line.strip() == title) + 1
    header = [h.strip() for h in lines[start].split(",")]
    rows = []
    for line in lines[start + 1:]:
        fields = [f.strip() for f in line.split(",")]
        if len(fields[0]) != key_width or not fields[0].isdigit():
            break
        if len(fields) != len(header):
            raise ValueError(f"ragged row {fields[0]}")
        for token in fields[1:]:
            float(token)
        rows.append(fields)
    if not rows:
        raise ValueError(f"empty table after {title!r}")
    return header, rows


def csv_text(key, header, rows):
    out = [",".join([key] + header)]
    out += [",".join(row) for row in rows]
    return "\n".join(out) + "\n"


def _french(source, title=None, key="yyyymm", key_width=6, start=None):
    def derive(payloads):
        header, rows = french_section(payloads[source], title, key_width)
        if start is not None:
            rows = [row for row in rows if row[0] >= start]
        return csv_text(key, header[1:], rows)
    return derive


_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _column_index(reference):
    letters = re.match(r"[A-Z]+", reference).group(0)
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - 64
    return index - 1


def xlsx_sheet(payload, name):
    """Cell text of one worksheet of an .xlsx file (no third-party reader needed)."""
    archive = zipfile.ZipFile(io.BytesIO(payload))
    shared = []
    if "xl/sharedStrings.xml" in archive.namelist():
        for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).iter(_SHEET_NS + "si"):
            shared.append("".join(node.text or "" for node in item.iter(_SHEET_NS + "t")))
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relations = {r.get("Id"): r.get("Target") for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
    sheet = next(s for s in workbook.iter(_SHEET_NS + "sheet") if s.get("name") == name)
    target = relations[sheet.get(_REL_NS + "id")].lstrip("/")
    target = target if target.startswith("xl/") else "xl/" + target
    rows = []
    for row in ET.fromstring(archive.read(target)).iter(_SHEET_NS + "row"):
        cells = {}
        for cell in row.iter(_SHEET_NS + "c"):
            value = cell.find(_SHEET_NS + "v")
            if value is None:
                inline = cell.find(_SHEET_NS + "is")
                text = "".join(n.text or "" for n in inline.iter(_SHEET_NS + "t")) if inline is not None else None
            else:
                text = shared[int(value.text)] if cell.get("t") == "s" else value.text
            cells[_column_index(cell.get("r"))] = text
        if cells:
            rows.append([cells.get(i) for i in range(max(cells) + 1)])
    return rows


def goyal_welch_monthly(payloads):
    """Monthly sheet of the updated Welch-Goyal predictor file; empty cells become NaN."""
    rows = xlsx_sheet(payloads["goyal-welch-predictors-2025"], "Monthly")
    header = [h.strip() for h in rows[0]]
    if header[0] != "yyyymm" or "CRSP_SPvw" not in header:
        raise ValueError(f"unexpected Welch-Goyal header {header}")
    body = []
    for row in rows[1:]:
        row = (row + [None] * len(header))[:len(header)]
        tokens = [str(int(float(row[0])))] + [("NaN" if v in (None, "") else v.strip()) for v in row[1:]]
        for token in tokens[1:]:
            float(token)
        body.append(tokens)
    return csv_text("yyyymm", header[1:], body)


def quantlib_dax_surface(payloads):
    """DAX implied volatilities of 2002-07-05 (A. Sepp's data) as embedded in QuantLib's Heston tests."""
    source = payloads["quantlib-heston-test"].decode("utf-8")
    start = source.index("CalibrationMarketData getDAXCalibrationMarketData()")
    block = source[start:source.index("struct HestonProcessDiscretizationDesc", start)]

    def array(name):
        body = re.search(name + r"\[\]\s*=\s*\{([^}]*)\}", block).group(1)
        return re.findall(r"[0-9]+(?:\.[0-9]+)?", body)
    days, rates, vols, strikes = array("Integer t"), array("Rate r"), array("Volatility v"), array("Real strike")
    spot = re.search(r"SimpleQuote>\(([0-9.]+)\)\);\s*Real strike", block).group(1)
    if (len(days), len(rates), len(vols), len(strikes)) != (8, 8, 104, 13):
        raise ValueError("unexpected DAX surface layout")
    rows = [[f"K{strike}-D{day}", strike, day, rate, vols[s * 8 + m], spot]
            for s, strike in enumerate(strikes) for m, (day, rate) in enumerate(zip(days, rates))]
    return csv_text("quote", ["strike", "maturity_days", "zero_rate", "implied_vol", "spot"], rows)


KIDNEY_POOLS = {"00036-00000151": 256, "00036-00000152": 256, "00036-00000153": 256, "00036-00000154": 256,
                "00036-00000191": 512, "00036-00000192": 512, "00036-00000193": 512, "00036-00000194": 512}


def preflib_kidney_pools(payloads):
    """Compatibility edges (donor of pair a can give to patient of pair b) of eight PrefLib kidney pools."""
    rows = []
    for name, pairs in KIDNEY_POOLS.items():
        text = payloads["preflib-kidney-" + name].decode("utf-8")
        header = re.search(r"^# NUMBER ALTERNATIVES: (\d+)$", text, re.M)
        if header is None or int(header.group(1)) != pairs or "with 0\n" not in text:
            raise ValueError(f"{name}: expected a pool of {pairs} pairs without altruists")
        for line in text.splitlines():
            if line and not line.startswith("#"):
                donor, patient, weight = line.split(",")
                if float(weight) != 1.0 or not (1 <= int(donor) <= pairs and 1 <= int(patient) <= pairs):
                    raise ValueError(f"{name}: unexpected edge {line}")
                rows.append([name, donor, patient])
    return csv_text("instance", ["donor_pair", "patient_pair"], rows)


def gsw_monthly(payloads):
    """Last daily observation of each complete month of the GSW zero-coupon curve, maturities 1-10 years (percent)."""
    lines = payloads["fed-gsw-yield-curve"].decode("utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    header = lines[start].split(",")
    wanted = [f"SVENY{m:02d}" for m in range(1, 11)]
    index = [header.index(name) for name in wanted]
    last = {}
    for line in lines[start + 1:]:
        fields = line.split(",")
        values = [fields[j] for j in index]
        if "NA" in values or "" in values:
            continue
        for token in values:
            float(token)
        last[fields[0][:4] + fields[0][5:7]] = [fields[0]] + values
    months = sorted(last)
    final_day = max(row[0] for row in last.values())
    if final_day[8:] < "28":  # drop a trailing month with only a few days of data
        months = [m for m in months if m != final_day[:4] + final_day[5:7]]
    rows = [[m, last[m][0].replace("-", "")] + last[m][1:] for m in months if m >= "197201"]
    return csv_text("yyyymm", ["yyyymmdd"] + wanted, rows)


def fred_md(payloads):
    """FRED-MD vintage: a 'tcode' row with the transformation codes, then one row per month (yyyymm keys)."""
    import csv
    rows = [r for r in csv.reader(io.StringIO(payloads["fred-md-2024-10"].decode("utf-8"))) if r and r[0]]
    header, codes = rows[0], rows[1]
    if header[0] != "sasdate" or codes[0] != "Transform:" or "INDPRO" not in header:
        raise ValueError("unexpected FRED-MD layout")
    body = []
    for row in rows[2:]:
        month, _, year = row[0].split("/")
        tokens = [token.strip() for token in row[1:]]
        for token in tokens:
            if token:
                float(token)
        body.append([f"{int(year):04d}{int(month):02d}"] + tokens)
    return csv_text("yyyymm", header[1:], [["tcode"] + [str(int(float(c))) for c in codes[1:]]] + body)


def orlib_port(index):
    """OR-Library portN: mean and standard deviation of weekly returns and the full correlation matrix."""
    def derive(payloads):
        tokens = payloads[f"orlib-port{index}"].decode("ascii").split()
        n = int(tokens[0])
        stats = [tokens[1 + 2 * a: 3 + 2 * a] for a in range(n)]
        corr = [[None] * n for _ in range(n)]
        rest = tokens[1 + 2 * n:]
        for k in range(0, len(rest), 3):
            i, j, value = int(rest[k]) - 1, int(rest[k + 1]) - 1, rest[k + 2]
            float(value)
            corr[i][j] = corr[j][i] = value
        if any(v is None for row in corr for v in row):
            raise ValueError(f"port{index}: incomplete correlation matrix")
        rows = [[str(a + 1)] + stats[a] + corr[a] for a in range(n)]
        return csv_text("asset", ["mean", "sd"] + [f"c{j + 1}" for j in range(n)], rows)
    return derive


def orlib_portef(index):
    """OR-Library portefN: the published unconstrained efficient frontier (mean return, variance) points."""
    def derive(payloads):
        tokens = payloads[f"orlib-portef{index}"].decode("ascii").split()
        rows = [[str(k + 1), tokens[2 * k], tokens[2 * k + 1]] for k in range(len(tokens) // 2)]
        for row in rows:
            float(row[1]), float(row[2])
        return csv_text("point", ["mean_return", "variance"], rows)
    return derive


def orlib_indtrack(index):
    """OR-Library indtrackN: weekly index and stock prices for weeks 0..T (one row per week)."""
    def derive(payloads):
        tokens = payloads[f"orlib-indtrack{index}"].decode("ascii").split()
        n, periods = int(tokens[0]), int(tokens[1])
        values = tokens[2:]
        if len(values) != (n + 1) * (periods + 1):
            raise ValueError(f"indtrack{index}: expected {(n + 1) * (periods + 1)} prices, found {len(values)}")
        series = [values[s * (periods + 1):(s + 1) * (periods + 1)] for s in range(n + 1)]
        rows = [[str(t)] + [series[s][t] for s in range(n + 1)] for t in range(periods + 1)]
        for row in rows:
            if min(float(x) for x in row[1:]) <= 0:
                raise ValueError(f"indtrack{index}: nonpositive price")
        return csv_text("week", ["index"] + [f"s{i + 1}" for i in range(n)], rows)
    return derive


VW_MONTHLY = "Average Value Weighted Returns -- Monthly"
VW_DAILY = "Average Value Weighted Returns -- Daily"

DERIVERS = {
    "ff_factors_monthly.csv": _french("french-ff3-factors-monthly"),
    "ind10_vw_monthly.csv": _french("french-10-industries-monthly", VW_MONTHLY),
    "ff25_vw_monthly.csv": _french("french-25-size-bm-monthly", VW_MONTHLY),
    "ind49_vw_monthly.csv": _french("french-49-industries-monthly", VW_MONTHLY),
    "goyal_welch_monthly.csv": goyal_welch_monthly,
    "dax_2002_surface.csv": quantlib_dax_surface,
    "kidney_preflib_pools.csv": preflib_kidney_pools,
    "ff_factors_daily.csv": _french("french-ff3-factors-daily", key="yyyymmdd", key_width=8),
    "gsw_yields_monthly.csv": gsw_monthly,
    "fredmd_2024_10.csv": fred_md,
    **{f"orlib_port{i}.csv": orlib_port(i) for i in range(1, 6)},
    **{f"orlib_portef{i}.csv": orlib_portef(i) for i in range(1, 6)},
    **{f"orlib_indtrack{i}.csv": orlib_indtrack(i) for i in range(1, 6)},
    "ind49_vw_daily.csv": _french("french-49-industries-daily", VW_DAILY, key="yyyymmdd", key_width=8, start="19710312"),
}
