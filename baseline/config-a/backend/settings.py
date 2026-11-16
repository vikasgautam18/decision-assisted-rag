"""Anonymous sandbox settings. Environment variables override .env."""
import os
from pathlib import Path

from common import config as _C  # loads .env

ROOT = Path(__file__).resolve().parent.parent
AUDIT_FILE = Path(os.getenv("AUDIT_FILE", ROOT / "logs" / "audit.jsonl"))
MAX_HISTORY_TURNS = _C.env_int("MAX_HISTORY_TURNS", 10)
EXCEL_TTL_SECONDS = _C.env_int("EXCEL_TTL_SECONDS", 2 * 3600)
EXCEL_MAX_BYTES = _C.env_int("EXCEL_MAX_BYTES", 25 * 1024 * 1024)
EXCEL_MAX_QUESTIONS = _C.env_int("EXCEL_MAX_QUESTIONS", 100)  # per workbook session
