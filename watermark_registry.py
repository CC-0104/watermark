"""SQLite-backed registry for TrustMark watermark IDs.

The image watermark stores only a short ID. This registry maps that ID to the
full business payload and audit metadata.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().with_name(".wm_trustmark_registry.sqlite")
ID_LENGTH = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WatermarkRegistry:
    """Small repository layer for watermark ID lookup records."""

    def __init__(self, db_path: str | os.PathLike[str] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS watermark_records (
                    watermark_id TEXT PRIMARY KEY,
                    payload_text TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT,
                    source_image_hash TEXT,
                    algorithm TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner TEXT,
                    note TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_watermark_payload_hash ON watermark_records(payload_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_watermark_status ON watermark_records(status)"
            )

    def _candidate_ids(self, payload_text: str, source_image_hash: str = ""):
        seed = f"{payload_text}\n{source_image_hash}".encode("utf-8")
        digest = hashlib.sha256(seed).digest()
        stream = base64.b32encode(digest).decode("ascii").rstrip("=")
        for offset in range(0, len(stream) - ID_LENGTH + 1):
            yield stream[offset: offset + ID_LENGTH]

    def register(
        self,
        payload_text: str,
        *,
        source_image_hash: str = "",
        algorithm: str = "trustmark",
        payload_json: dict[str, Any] | None = None,
        owner: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        payload_hash = sha256_text(payload_text)
        payload_json_text = json.dumps(payload_json, ensure_ascii=False, sort_keys=True) if payload_json else None
        now = utc_now()

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM watermark_records
                WHERE payload_hash = ? AND COALESCE(source_image_hash, '') = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (payload_hash, source_image_hash),
            ).fetchone()
            if existing:
                return dict(existing)

            for watermark_id in self._candidate_ids(payload_text, source_image_hash):
                collision = conn.execute(
                    "SELECT payload_hash FROM watermark_records WHERE watermark_id = ?",
                    (watermark_id,),
                ).fetchone()
                if collision and collision["payload_hash"] != payload_hash:
                    continue

                conn.execute(
                    """
                    INSERT OR REPLACE INTO watermark_records (
                        watermark_id, payload_text, payload_hash, payload_json,
                        source_image_hash, algorithm, status, created_at,
                        updated_at, owner, note
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
                    """,
                    (
                        watermark_id,
                        payload_text,
                        payload_hash,
                        payload_json_text,
                        source_image_hash,
                        algorithm,
                        now,
                        now,
                        owner,
                        note,
                    ),
                )
                return self.get(watermark_id) or {"watermark_id": watermark_id}

        raise ValueError("Unable to allocate a unique watermark_id")

    def get(self, watermark_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM watermark_records WHERE watermark_id = ?",
                (watermark_id,),
            ).fetchone()
        return dict(row) if row else None

    def resolve(self, watermark_id: str) -> str | None:
        record = self.get(watermark_id)
        if not record or record.get("status") != "active":
            return None
        return record["payload_text"]

    def update_status(self, watermark_id: str, status: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE watermark_records SET status = ?, updated_at = ? WHERE watermark_id = ?",
                (status, utc_now(), watermark_id),
            )
            return cur.rowcount > 0

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM watermark_records ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
