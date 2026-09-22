"""SQLite log of questions and feedback. The unanswered-questions report shows where the documents have gaps."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session_id TEXT, question TEXT, standalone TEXT,
    mode TEXT, answered INTEGER, best_score REAL, latency_ms REAL, sources TEXT, answer TEXT);
CREATE TABLE IF NOT EXISTS feedback(
    id INTEGER PRIMARY KEY AUTOINCREMENT, query_id INTEGER, ts TEXT, rating INTEGER, comment TEXT);
"""


class Store:
    def __init__(self, path):
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _rows(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def log_query(self, *, session_id, question, standalone, mode, answered, best_score, latency_ms, sources, answer) -> int:
        best = best_score if best_score not in (float("-inf"), float("inf")) else None
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO queries(ts,session_id,question,standalone,mode,answered,best_score,latency_ms,sources,answer)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (self._now(), session_id, question, standalone, mode, int(answered), best, latency_ms,
                 json.dumps(sources), answer))
            self._db.commit()
            return int(cur.lastrowid)

    def add_feedback(self, query_id: int, rating: int, comment: str = "") -> None:
        if rating not in (-1, 1):
            raise ValueError("rating must be 1 or -1")
        with self._lock:
            if not self._db.execute("SELECT 1 FROM queries WHERE id=?", (query_id,)).fetchone():
                raise KeyError(f"unknown query_id {query_id}")
            self._db.execute("INSERT INTO feedback(query_id,ts,rating,comment) VALUES(?,?,?,?)",
                             (query_id, self._now(), rating, comment[:500]))
            self._db.commit()

    # ---- reports
    def unanswered(self, limit: int = 50):
        """Questions the assistant declined to answer, most frequent first."""
        return self._rows(
            "SELECT LOWER(TRIM(question)) AS question, COUNT(*) AS times, MAX(ts) AS last_asked "
            "FROM queries WHERE answered=0 GROUP BY LOWER(TRIM(question)) ORDER BY times DESC, last_asked DESC LIMIT ?",
            (limit,))

    def downvoted(self, limit: int = 50):
        return self._rows(
            "SELECT q.id, q.question, q.answer, f.comment, f.ts FROM feedback f JOIN queries q ON q.id=f.query_id "
            "WHERE f.rating=-1 ORDER BY f.ts DESC LIMIT ?", (limit,))

    def top_questions(self, limit: int = 20):
        return self._rows(
            "SELECT LOWER(TRIM(question)) AS question, COUNT(*) AS times FROM queries "
            "GROUP BY LOWER(TRIM(question)) ORDER BY times DESC LIMIT ?", (limit,))

    def stats(self) -> dict:
        q = self._rows("SELECT COUNT(*) AS n, COALESCE(AVG(answered),0) AS answered_rate, "
                       "COALESCE(AVG(latency_ms),0) AS avg_latency_ms FROM queries")[0]
        f = self._rows("SELECT COALESCE(SUM(rating=1),0) AS up, COALESCE(SUM(rating=-1),0) AS down FROM feedback")[0]
        modes = {r["mode"]: r["n"] for r in self._rows("SELECT mode, COUNT(*) AS n FROM queries GROUP BY mode")}
        return {"questions": q["n"], "answered_rate": round(q["answered_rate"], 3),
                "avg_latency_ms": round(q["avg_latency_ms"], 1), "thumbs_up": f["up"], "thumbs_down": f["down"],
                "by_mode": modes}
