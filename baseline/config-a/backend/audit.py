"""Operational JSON-lines events; no questions, workbook content, or session credentials."""
import json
import time

from backend import settings as S


def record(**event):
    S.AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with S.AUDIT_FILE.open("a") as f:
        f.write(json.dumps(event) + "\n")
