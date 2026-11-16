from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
import uuid
from pathlib import Path


class ConversationNotFound(LookupError):
    pass


class ConversationBusy(RuntimeError):
    pass


class SessionStore:
    def __init__(self, path: Path, *, ttl_seconds: int = 1800) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.key = secrets.token_bytes(32)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS conversations "
                "(id TEXT PRIMARY KEY, owner TEXT NOT NULL, updated REAL NOT NULL, "
                "busy INTEGER NOT NULL, messages TEXT NOT NULL)"
            )
            connection.execute("UPDATE conversations SET busy=0")
        path.chmod(0o600)

    def session(self, cookie: str) -> tuple[str, str]:
        if re.fullmatch(r"[a-f0-9]{64}\.[a-f0-9]{64}", cookie):
            sid, signature = cookie.split(".")
            expected = hmac.new(self.key, sid.encode(), hashlib.sha256).hexdigest()
            if hmac.compare_digest(signature, expected):
                return sid, cookie
        sid = secrets.token_hex(32)
        signature = hmac.new(self.key, sid.encode(), hashlib.sha256).hexdigest()
        return sid, f"{sid}.{signature}"

    def begin(self, owner: str, conversation_id: str | None) -> tuple[str, list[dict]]:
        now = time.time()
        with sqlite3.connect(self.path, isolation_level=None) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM conversations WHERE updated < ? AND busy=0",
                (now - self.ttl_seconds,),
            )
            if conversation_id is None:
                count = connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
                if count >= 128:
                    raise ConversationBusy("pilot conversation capacity is reached")
                conversation_id = uuid.uuid4().hex
                connection.execute(
                    "INSERT INTO conversations VALUES (?,?,?,0,'[]')",
                    (conversation_id, owner, now),
                )
            row = connection.execute(
                "SELECT owner,busy,messages FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
            if row is None or row[0] != owner:
                raise ConversationNotFound("conversation not found")
            if row[1]:
                raise ConversationBusy("a request for this conversation is already running")
            connection.execute(
                "UPDATE conversations SET busy=1,updated=? WHERE id=?", (now, conversation_id)
            )
            connection.execute("COMMIT")
        return conversation_id, json.loads(row[2])

    def finish(self, owner: str, conversation_id: str, question: str, result: dict | None) -> None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT messages FROM conversations WHERE id=? AND owner=?",
                (conversation_id, owner),
            ).fetchone()
            if row is None:
                raise ConversationNotFound("conversation not found")
            messages = json.loads(row[0])
            if result is not None and result["status"] == "completed":
                messages.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": result["answer"]},
                    ]
                )
            connection.execute(
                "UPDATE conversations SET busy=0,updated=?,messages=? WHERE id=? AND owner=?",
                (time.time(), json.dumps(messages[-16:]), conversation_id, owner),
            )

    def history(self, owner: str, conversation_id: str) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT owner,updated,messages FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
        if row is None or row[0] != owner or row[1] < time.time() - self.ttl_seconds:
            raise ConversationNotFound("conversation not found")
        return json.loads(row[2])
