"""
Persistent storage for child progress: every attempt at every word,
line, and stanza; which lessons are unlocked; a daily practice streak;
and now the EXACT position within a lesson (stanza, word, and phase),
so a child who stops partway through and comes back later resumes
exactly where they left off, not from the start of the stanza.

SQLite specifically because Streamlit re-runs the whole script on every
interaction -- anything that needs to survive between reruns (and
between separate app launches, so both a streak and a resume position
actually mean something) has to live outside Python's in-memory state.
"""
import sqlite3
import os
from datetime import date, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "progress.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS children (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT (date('now'))
        );

        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER NOT NULL REFERENCES children(id),
            lesson_id TEXT NOT NULL,
            stanza_index INTEGER NOT NULL,
            unit TEXT NOT NULL,          -- 'word' or 'stanza' or 'poem'
            unit_label TEXT NOT NULL,    -- the actual word or full line text
            score REAL,
            passed INTEGER NOT NULL,
            attempted_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS lesson_progress (
            child_id INTEGER NOT NULL REFERENCES children(id),
            lesson_id TEXT NOT NULL,
            stanza_index INTEGER NOT NULL DEFAULT 0,
            word_index INTEGER NOT NULL DEFAULT 0,
            phase TEXT NOT NULL DEFAULT 'word',
            lesson_completed INTEGER NOT NULL DEFAULT 0,
            completed_at TEXT,
            PRIMARY KEY (child_id, lesson_id)
        );

        CREATE TABLE IF NOT EXISTS practice_days (
            child_id INTEGER NOT NULL REFERENCES children(id),
            practice_date TEXT NOT NULL,
            PRIMARY KEY (child_id, practice_date)
        );

        -- EXTENSION: Duolingo-style XP. A real, persistent point total,
        -- not a decorative number -- every row here is one specific
        -- passed check (a word, a stanza, a whole poem, a Custom
        -- Practice sentence), so the total is always traceable back to
        -- real attempts in the `attempts` table, the same honesty
        -- principle as every score already stored in this project.
        CREATE TABLE IF NOT EXISTS xp_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER NOT NULL REFERENCES children(id),
            amount INTEGER NOT NULL,
            reason TEXT NOT NULL,        -- 'word' / 'stanza' / 'poem' / 'sentence'
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(lesson_progress)")}
    if "word_index" not in existing_cols:
        conn.execute("ALTER TABLE lesson_progress ADD COLUMN word_index INTEGER NOT NULL DEFAULT 0")
    if "phase" not in existing_cols:
        conn.execute("ALTER TABLE lesson_progress ADD COLUMN phase TEXT NOT NULL DEFAULT 'word'")
    conn.commit()
    conn.close()


def get_or_create_child(name):
    name = name.strip()
    conn = get_connection()
    row = conn.execute("SELECT id FROM children WHERE name = ?", (name,)).fetchone()
    if row:
        child_id = row["id"]
    else:
        cur = conn.execute("INSERT INTO children (name) VALUES (?)", (name,))
        conn.commit()
        child_id = cur.lastrowid
    conn.close()
    return child_id


def list_children():
    """Every child (student) who has ever practiced, by name -- used by
    the teacher-facing "All Students" view. Ordered by name, not
    creation order, so a growing class list stays easy to scan."""
    conn = get_connection()
    rows = conn.execute("SELECT id, name FROM children ORDER BY name COLLATE NOCASE").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_attempt(child_id, lesson_id, stanza_index, unit, unit_label, score, passed):
    conn = get_connection()
    conn.execute(
        "INSERT INTO attempts (child_id, lesson_id, stanza_index, unit, unit_label, score, passed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (child_id, lesson_id, stanza_index, unit, unit_label, score, int(passed)),
    )
    conn.execute(
        "INSERT OR IGNORE INTO practice_days (child_id, practice_date) VALUES (?, date('now'))",
        (child_id,),
    )
    conn.commit()
    conn.close()


# EXTENSION: Duolingo-style XP amounts -- deliberately reflects real
# effort (a whole poem is worth far more than one word), same spirit as
# Duolingo's own "harder exercises are worth more XP." Kept as a single
# named constant, not scattered magic numbers, so the values are easy
# to find and tune later.
XP_VALUES = {"word": 10, "stanza": 25, "poem": 60, "sentence": 15}


def award_xp(child_id, reason):
    """Logs one real XP event for a passed check and returns the
    amount awarded, or 0 for an unrecognized reason (never crashes the
    calling check endpoint over a game-mechanic add-on)."""
    amount = XP_VALUES.get(reason, 0)
    if amount <= 0:
        return 0
    conn = get_connection()
    conn.execute(
        "INSERT INTO xp_events (child_id, amount, reason) VALUES (?, ?, ?)",
        (child_id, amount, reason),
    )
    conn.commit()
    conn.close()
    return amount


def get_total_xp(child_id):
    conn = get_connection()
    row = conn.execute("SELECT COALESCE(SUM(amount), 0) t FROM xp_events WHERE child_id = ?", (child_id,)).fetchone()
    conn.close()
    return row["t"]


def get_lesson_progress(child_id, lesson_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM lesson_progress WHERE child_id = ? AND lesson_id = ?",
        (child_id, lesson_id),
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        "child_id": child_id, "lesson_id": lesson_id,
        "stanza_index": 0, "word_index": 0, "phase": "word",
        "lesson_completed": 0, "completed_at": None,
    }


def set_stanza_index(child_id, lesson_id, stanza_index):
    conn = get_connection()
    conn.execute(
        "INSERT INTO lesson_progress (child_id, lesson_id, stanza_index) VALUES (?, ?, ?) "
        "ON CONFLICT(child_id, lesson_id) DO UPDATE SET stanza_index = excluded.stanza_index",
        (child_id, lesson_id, stanza_index),
    )
    conn.commit()
    conn.close()


def set_progress_position(child_id, lesson_id, stanza_index, word_index, phase):
    conn = get_connection()
    conn.execute(
        "INSERT INTO lesson_progress (child_id, lesson_id, stanza_index, word_index, phase) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(child_id, lesson_id) DO UPDATE SET "
        "stanza_index = excluded.stanza_index, word_index = excluded.word_index, phase = excluded.phase",
        (child_id, lesson_id, stanza_index, word_index, phase),
    )
    conn.commit()
    conn.close()


def mark_lesson_completed(child_id, lesson_id):
    conn = get_connection()
    conn.execute(
        "INSERT INTO lesson_progress (child_id, lesson_id, lesson_completed, completed_at) "
        "VALUES (?, ?, 1, datetime('now')) "
        "ON CONFLICT(child_id, lesson_id) DO UPDATE SET lesson_completed = 1, completed_at = datetime('now')",
        (child_id, lesson_id),
    )
    conn.commit()
    conn.close()


def is_lesson_unlocked(child_id, lesson_id, lang_lessons):
    idx = next((i for i, l in enumerate(lang_lessons) if l["id"] == lesson_id), None)
    if idx is None or idx == 0:
        return True
    prev_lesson = lang_lessons[idx - 1]
    prog = get_lesson_progress(child_id, prev_lesson["id"])
    return bool(prog["lesson_completed"])


def get_streak(child_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT practice_date FROM practice_days WHERE child_id = ? ORDER BY practice_date DESC",
        (child_id,),
    ).fetchall()
    conn.close()
    if not rows:
        return 0
    practiced = {date.fromisoformat(r["practice_date"]) for r in rows}
    today = date.today()
    if today not in practiced and (today - timedelta(days=1)) not in practiced:
        return 0
    streak = 0
    cursor = today if today in practiced else today - timedelta(days=1)
    while cursor in practiced:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def get_language_stats(child_id, lang_lessons):
    lesson_ids = [l["id"] for l in lang_lessons]
    if not lesson_ids:
        return {"attempts": 0, "avg_score": None, "completed": 0, "total": 0}
    placeholders = ",".join("?" * len(lesson_ids))
    conn = get_connection()
    row = conn.execute(
        f"SELECT COUNT(*) c, AVG(score) a FROM attempts "
        f"WHERE child_id = ? AND lesson_id IN ({placeholders}) AND score IS NOT NULL",
        (child_id, *lesson_ids),
    ).fetchone()
    completed = conn.execute(
        f"SELECT COUNT(*) c FROM lesson_progress "
        f"WHERE child_id = ? AND lesson_completed = 1 AND lesson_id IN ({placeholders})",
        (child_id, *lesson_ids),
    ).fetchone()["c"]
    conn.close()
    return {
        "attempts": row["c"],
        "avg_score": round(row["a"], 1) if row["a"] else None,
        "completed": completed,
        "total": len(lesson_ids),
    }


def get_dashboard_stats(child_id):
    conn = get_connection()
    total_attempts = conn.execute(
        "SELECT COUNT(*) c FROM attempts WHERE child_id = ?", (child_id,)
    ).fetchone()["c"]
    word_attempts = conn.execute(
        "SELECT COUNT(*) c FROM attempts WHERE child_id = ? AND unit = 'word'", (child_id,)
    ).fetchone()["c"]
    stanza_attempts = conn.execute(
        "SELECT COUNT(*) c FROM attempts WHERE child_id = ? AND unit = 'stanza'", (child_id,)
    ).fetchone()["c"]
    avg_score = conn.execute(
        "SELECT AVG(score) a FROM attempts WHERE child_id = ? AND score IS NOT NULL", (child_id,)
    ).fetchone()["a"]
    lessons_completed = conn.execute(
        "SELECT COUNT(*) c FROM lesson_progress WHERE child_id = ? AND lesson_completed = 1", (child_id,)
    ).fetchone()["c"]
    recent = conn.execute(
        "SELECT lesson_id, unit, unit_label, score, passed, attempted_at FROM attempts "
        "WHERE child_id = ? ORDER BY attempted_at DESC LIMIT 15",
        (child_id,),
    ).fetchall()
    daily_counts = conn.execute(
        "SELECT substr(attempted_at,1,10) as d, COUNT(*) as n FROM attempts WHERE child_id = ? "
        "GROUP BY d ORDER BY d DESC LIMIT 14",
        (child_id,),
    ).fetchall()
    conn.close()
    return {
        "total_attempts": total_attempts,
        "word_attempts": word_attempts,
        "stanza_attempts": stanza_attempts,
        "avg_score": round(avg_score, 1) if avg_score else None,
        "lessons_completed": lessons_completed,
        "recent": [dict(r) for r in recent],
        "daily_counts": [dict(r) for r in daily_counts],
        "streak": get_streak(child_id),
    }


def get_all_children_summary():
    conn = get_connection()
    children = conn.execute("SELECT id, name FROM children ORDER BY name").fetchall()
    rows = []
    for c in children:
        cid = c["id"]
        stats = conn.execute(
            "SELECT COUNT(*) attempts, AVG(score) avg_score, "
            "MAX(attempted_at) last_practiced FROM attempts WHERE child_id = ?",
            (cid,),
        ).fetchone()
        lessons_completed = conn.execute(
            "SELECT COUNT(*) c FROM lesson_progress WHERE child_id = ? AND lesson_completed = 1",
            (cid,),
        ).fetchone()["c"]
        rows.append({
            "Name": c["name"],
            "Recorded Voice?": "✅ Yes" if stats["attempts"] > 0 else "— Not yet",
            "Total Attempts": stats["attempts"],
            "Avg Score": f"{round(stats['avg_score'], 1)}" if stats["avg_score"] is not None else "—",
            "Lessons Completed": lessons_completed,
            "Current Streak": get_streak(cid),
            "Last Practiced": stats["last_practiced"] if stats["last_practiced"] else "Never",
        })
    conn.close()
    return rows


# ============================================================================
# EXTENSION: teacher-uploaded lesson content (PDF ingestion feature)
#
# The 3 built-in lessons in modules/content.py are a fixed Python list --
# fine for hand-verified, permanent curriculum content, but a teacher
# uploading their own PDF at runtime needs somewhere to persist that new
# content between app restarts, which a hardcoded list can't do. This
# table is exactly that: one row per teacher-uploaded lesson, storing
# its stanzas as JSON text (same shape as a built-in lesson's "stanzas"
# list, so the rest of the app -- word practice, stanza/poem checks,
# resume, gating -- treats an uploaded lesson identically to a built-in
# one without needing to know the difference).
#
# Deliberately a SEPARATE table from lesson_progress: this table is the
# CONTENT of a lesson (what a teacher uploaded), while lesson_progress
# is a CHILD's position within any lesson (built-in or uploaded) --
# keeping these separate means nothing about how progress is tracked
# needed to change for this feature at all.
# ============================================================================
import json as _json


def _init_uploaded_lessons_table():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS uploaded_lessons (
            id TEXT PRIMARY KEY,
            language TEXT NOT NULL,
            language_label TEXT NOT NULL,
            title TEXT NOT NULL,
            subtitle TEXT NOT NULL,
            stanzas_json TEXT NOT NULL,
            uploaded_by TEXT,
            source_filename TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    conn.close()


def add_uploaded_lesson(lesson_id, language, language_label, title, subtitle, stanzas, uploaded_by=None, source_filename=None):
    """
    Saves one teacher-uploaded lesson. `stanzas` is a list of stanzas,
    each stanza itself a list of line strings -- exactly the same shape
    used by the built-in LESSONS in modules/content.py. Raises
    sqlite3.IntegrityError if lesson_id already exists (caller should
    generate a fresh, collision-checked id -- see content.py).
    """
    _init_uploaded_lessons_table()
    conn = get_connection()
    conn.execute(
        "INSERT INTO uploaded_lessons "
        "(id, language, language_label, title, subtitle, stanzas_json, uploaded_by, source_filename) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (lesson_id, language, language_label, title, subtitle, _json.dumps(stanzas), uploaded_by, source_filename),
    )
    conn.commit()
    conn.close()


def get_uploaded_lessons():
    """
    Every teacher-uploaded lesson, converted to the exact same dict
    shape as a built-in lesson from modules/content.py (id, language,
    language_label, title, subtitle, image, stanzas) -- so content.py
    can merge these into its existing lesson lists without the rest of
    the app (app.py's practice/gating/resume logic) needing to treat
    them as a special case at all.
    """
    _init_uploaded_lessons_table()
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM uploaded_lessons ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    lessons = []
    for r in rows:
        lessons.append({
            "id": r["id"],
            "language": r["language"],
            "language_label": r["language_label"],
            "title": r["title"],
            "subtitle": r["subtitle"],
            "image": None,  # uploaded lessons have no illustration
            "stanzas": _json.loads(r["stanzas_json"]),
            "is_uploaded": True,
        })
    return lessons


def lesson_id_exists(lesson_id):
    """Checks BOTH uploaded lessons and needs to also be checked
    against built-in ids by the caller (content.py) -- this function
    only knows about the uploaded_lessons table specifically."""
    _init_uploaded_lessons_table()
    conn = get_connection()
    row = conn.execute("SELECT 1 FROM uploaded_lessons WHERE id = ?", (lesson_id,)).fetchone()
    conn.close()
    return row is not None


def delete_uploaded_lesson(lesson_id):
    """Lets a teacher remove a lesson they uploaded by mistake. Does
    NOT delete any child's attempts/progress against it (that stays as
    real history of what was attempted), matching how nothing else in
    this app deletes attempt history either."""
    conn = get_connection()
    conn.execute("DELETE FROM uploaded_lessons WHERE id = ?", (lesson_id,))
    conn.commit()
    conn.close()


# ============================================================================
# EXTENSION: Ravikumar-style dashboard data -- per-language score series
# for charting, and a "Needs Practice" (tricky words) list. Both are read
# straight from the real `attempts` table already populated by every
# word/stanza/poem check in app.py; nothing new is logged for this, it's
# just aggregated differently for the restyled dashboard.
# ============================================================================
def get_language_score_series(child_id, lesson_id_to_language, limit=25):
    """
    Real attempt history, newest first, tagged with which language each
    attempt's lesson belongs to -- the shape the dashboard's per-
    language chart needs (Kannada vs. English vs. Custom Practice,
    matching the three groups Ravikumar's own dashboard charts:
    Kannada/English/Custom Audio). `lesson_id_to_language` is a dict of
    {lesson_id: language_code} built by the caller from content.py, so
    this function stays a pure data query with no import of content.py.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT lesson_id, unit, unit_label, score, passed, attempted_at FROM attempts "
        "WHERE child_id = ? ORDER BY attempted_at DESC, id DESC LIMIT ?",
        (child_id, limit),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["language"] = lesson_id_to_language.get(d["lesson_id"], "custom")
        out.append(d)
    return out


def get_tricky_words(child_id, limit=8):
    """
    Real "Needs Practice" list -- individual WORDS (not whole
    stanzas/poems) this child has actually attempted more than once
    with a failing result at least once, ranked by lowest average
    score first. A word that has only ever been attempted once, or has
    always passed on the first try, doesn't show up here -- this is
    meant to surface genuine, repeated trouble spots, matching
    Ravikumar's own dashboard's tricky-words screen, using this
    project's real attempt history rather than any placeholder data.
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT unit_label, lesson_id,
               COUNT(*) attempts,
               SUM(passed) passes,
               AVG(score) avg_score
        FROM attempts
        WHERE child_id = ? AND unit = 'word'
        GROUP BY unit_label, lesson_id
        HAVING COUNT(*) > 1 AND SUM(passed) < COUNT(*)
        ORDER BY (CAST(SUM(passed) AS REAL) / COUNT(*)) ASC, attempts DESC
        LIMIT ?
        """,
        (child_id, limit),
    ).fetchall()
    conn.close()
    return [
        {
            "word": r["unit_label"],
            "lesson_id": r["lesson_id"],
            "attempts": r["attempts"],
            "passes": r["passes"],
            "avg_score": round(r["avg_score"], 1) if r["avg_score"] is not None else None,
        }
        for r in rows
    ]