"""
Sentinel Memory — SQLite-backed long-term memory.

Stores:
  - User preferences   (key → value)
  - Command history    (timestamped log)
  - Learned facts      (free-form key-value pairs)
  - Named locations    (project paths, custom dirs)
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DB_PATH_DEFAULT = "data/sentinel.db"

# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS preferences (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    command     TEXT NOT NULL,
    intent      TEXT NOT NULL,
    success     INTEGER NOT NULL DEFAULT 1,
    summary     TEXT,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    category    TEXT DEFAULT 'general',
    created_at  REAL NOT NULL,
    UNIQUE(key)
);

CREATE INDEX IF NOT EXISTS idx_history_intent ON history(intent);
CREATE INDEX IF NOT EXISTS idx_facts_key      ON facts(key);
"""


# ─────────────────────────────────────────────
# Memory class
# ─────────────────────────────────────────────

class SentinelMemory:
    """
    Thread-safe SQLite memory store.

    Quick start:
        mem = SentinelMemory()
        mem.set_preference("ide", "Rider")
        ide = mem.get_preference("ide")
        mem.log_command("open chrome", "open_app", success=True)
    """

    def __init__(self, db_path: str = DB_PATH_DEFAULT):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Initialisation ───────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── Preferences ──────────────────────────────────────────────────

    def set_preference(self, key: str, value: str) -> None:
        """Store / update a user preference."""
        key = key.strip().lower()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, str(value), time.time())
            )

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a preference by key."""
        key = key.strip().lower()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def all_preferences(self) -> Dict[str, str]:
        """Return all stored preferences."""
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM preferences ORDER BY key").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete_preference(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM preferences WHERE key = ?", (key.strip().lower(),))

    # ── Facts ─────────────────────────────────────────────────────────

    def remember(self, key: str, value: str, category: str = "general") -> None:
        """Store a named fact (upsert)."""
        key = key.strip().lower()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO facts (key, value, category, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, category=excluded.category",
                (key, str(value), category, time.time())
            )

    def recall(self, key: str) -> Optional[str]:
        """Retrieve a fact by key."""
        key = key.strip().lower()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def all_facts(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value, category, created_at FROM facts ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Smart lookup ─────────────────────────────────────────────────

    def fuzzy_recall(self, query: str) -> Optional[str]:
        """
        Try to find a fact matching the query loosely.
        Checks both key exact match and partial key match.
        """
        exact = self.recall(query)
        if exact:
            return exact

        q = query.strip().lower()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value FROM facts WHERE key LIKE ? LIMIT 5",
                (f"%{q}%",)
            ).fetchall()
        if rows:
            return rows[0]["value"]

        # Try preferences
        pref = self.get_preference(q)
        if pref:
            return pref

        return None

    def resolve_memory_token(self, token: str) -> Optional[str]:
        """
        Resolve special __remembered_X__ tokens from the classifier.
        Used by the planner/executor.
        """
        if token == "__remembered_ide__":
            return (
                self.recall("favourite ide")
                or self.recall("ide")
                or self.get_preference("ide")
            )
        if token == "__remembered_project__":
            return (
                self.recall("project")
                or self.recall("project folder")
                or self.get_preference("project")
            )
        return None

    # ── Command history ───────────────────────────────────────────────

    def log_command(
        self,
        command: str,
        intent:  str,
        success: bool  = True,
        summary: str   = "",
    ) -> None:
        """Record a command to history."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO history (command, intent, success, summary, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (command, intent, int(success), summary, time.time())
            )

    def recent_history(self, limit: int = 20) -> List[Dict]:
        """Retrieve the N most recent commands."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT command, intent, success, summary, created_at "
                "FROM history ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def history_by_intent(self, intent: str, limit: int = 10) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT command, success, summary, created_at "
                "FROM history WHERE intent = ? ORDER BY created_at DESC LIMIT ?",
                (intent, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> str:
        prefs  = self.all_preferences()
        facts  = self.all_facts()
        recent = self.recent_history(5)

        lines = ["═══ Sentinel Memory ═══"]
        if prefs:
            lines.append("Preferences:")
            for k, v in prefs.items():
                lines.append(f"  {k}: {v}")
        if facts:
            lines.append("Facts:")
            for f in facts:
                lines.append(f"  {f['key']}: {f['value']}")
        if recent:
            lines.append("Recent commands:")
            for h in recent:
                dt = datetime.fromtimestamp(h["created_at"]).strftime("%H:%M")
                ok = "✓" if h["success"] else "✗"
                lines.append(f"  {ok} [{dt}] {h['command']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        prefs = len(self.all_preferences())
        facts = len(self.all_facts())
        with self._conn() as conn:
            hist = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        return f"SentinelMemory(preferences={prefs}, facts={facts}, history={hist})"
