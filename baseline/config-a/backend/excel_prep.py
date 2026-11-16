"""Excel analyst: upload validation, LibreOffice recalculation and a deterministic workbook profile."""
import subprocess
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

MAX_BYTES = 25 * 1024 * 1024
MAX_UNCOMPRESSED = 300 * 1024 * 1024
PROFILE_MAX_SHEETS = 40
PROFILE_MAX_BLOCKS = 12


def validate(data: bytes, filename: str):
    if not filename.lower().endswith(".xlsx") or data[:2] != b"PK":
        raise ValueError("only .xlsx files are supported")
    if len(data) > MAX_BYTES:
        raise ValueError("file too large (25 MB limit)")
    try:
        with zipfile.ZipFile(BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > MAX_UNCOMPRESSED:
                raise ValueError("workbook expands to too much data")
            if "xl/workbook.xml" not in z.namelist():
                raise ValueError("not a valid .xlsx workbook")
    except zipfile.BadZipFile:
        raise ValueError("not a valid .xlsx workbook")


def recalculate(data: bytes):
    """Headless LibreOffice recalculation: workbooks written by libraries have no cached formula values.
    Returns (bytes, recalculated); falls back to the original when LibreOffice is missing or fails."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.xlsx"
        src.write_bytes(data)
        try:
            subprocess.run(["soffice", f"-env:UserInstallation=file://{d}/profile", "--headless", "--convert-to",
                            "xlsx:Calc MS Excel 2007 XML", "--outdir", str(Path(d) / "out"), str(src)],
                           check=True, capture_output=True, timeout=120)
            return (Path(d) / "out" / "in.xlsx").read_bytes(), True
        except Exception:
            return data, False


def _blocks(ws):
    """Row runs separated by fully blank rows: a cheap hint for several tables stacked on one sheet."""
    blocks, start, last = [], None, None
    for r in range(1, ws.max_row + 1):
        filled = any(ws.cell(r, c).value is not None for c in range(1, ws.max_column + 1))
        if filled and start is None:
            start = r
        if filled:
            last = r
        if not filled and start is not None:
            blocks.append((start, last))
            start = None
    if start is not None:
        blocks.append((start, last))
    return blocks


def profile(data: bytes) -> dict:
    """Sheets, used ranges, blank-row separated blocks with their first cells, and formula/cached-value counts."""
    wf = openpyxl.load_workbook(BytesIO(data), data_only=False)
    wv = openpyxl.load_workbook(BytesIO(data), data_only=True)
    sheets = []
    for ws in wf.worksheets[:PROFILE_MAX_SHEETS]:
        vs = wv[ws.title]
        formulas = uncached = 0
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str) and c.value.startswith("="):
                    formulas += 1
                    if vs[c.coordinate].value is None:
                        uncached += 1
        blocks = []
        for a, b in _blocks(ws)[:PROFILE_MAX_BLOCKS]:
            first = [str(ws.cell(a, c).value)[:30] for c in range(1, ws.max_column + 1) if ws.cell(a, c).value is not None]
            blocks.append({"rows": f"{a}-{b}", "first_row": first[:6]})
        sheets.append({"name": ws.title, "state": ws.sheet_state,
                       "used_range": f"A1:{get_column_letter(max(ws.max_column, 1))}{ws.max_row}",
                       "merged": len(ws.merged_cells.ranges), "formulas": formulas, "uncached_formulas": uncached,
                       "blocks": blocks})
    return {"sheets": sheets, "sheet_count": len(wf.worksheets)}


def profile_text(p: dict) -> str:
    lines = [f"Workbook profile ({p['sheet_count']} sheets; automatic, verify with code):"]
    for s in p["sheets"]:
        lines.append(f"- '{s['name']}' {s['used_range']}"
                     + (f" [{s['state']}]" if s["state"] != "visible" else "")
                     + f", {s['formulas']} formulas ({s['uncached_formulas']} without value), {s['merged']} merged ranges")
        for b in s["blocks"]:
            lines.append(f"    rows {b['rows']}: starts with {b['first_row']}")
    return "\n".join(lines)
