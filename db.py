"""
SQLite storage for the ReadyKomunidad chatbot.

Two tables live in a single file (chatbot.db, next to this script):

  chatlogs         - every user message and bot reply, grouped by session_id
  knowledge_base   - extensible climate Q&A the router can fall back to

The knowledge base is searched with simple keyword scoring (no extra
SQLite extensions required), so it works on every Python sqlite3 build.
Add new entries at runtime with add_knowledge(...).
"""

import os
import sqlite3
import re

# Keep the database file next to this script so it is easy to find/back up.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")


def _get_conn():
    """Open a fresh connection. Opening per-call is thread-safe for Flask."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the tables if they don't already exist."""
    conn = _get_conn()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chatlogs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,   -- 'user' or 'bot'
                message     TEXT    NOT NULL,
                lang        TEXT,                 -- detected language ('en'/'tl')
                topic       TEXT,                 -- router intent for the turn
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_chatlogs_session ON chatlogs(session_id);
            CREATE INDEX IF NOT EXISTS idx_chatlogs_created  ON chatlogs(created_at);

            CREATE TABLE IF NOT EXISTS knowledge_base (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category    TEXT    NOT NULL,
                question    TEXT    NOT NULL,   -- representative trigger question
                answer_en   TEXT    NOT NULL,   -- English answer
                answer_tl   TEXT,                -- Tagalog answer (optional)
                keywords    TEXT,                -- comma-separated match terms
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
    finally:
        conn.close()
    seed_knowledge_base()


def log_message(session_id, role, message, lang=None, topic=None):
    """Store one message (either 'user' or 'bot') for a conversation session."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO chatlogs (session_id, role, message, lang, topic) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, message, lang, topic),
        )
        conn.commit()
    finally:
        conn.close()


def search_knowledge_base(query, limit=3):
    """
    Score knowledge_base rows by how many query words appear in the
    question/keywords/answer. Returns a list of dicts (best first).

    Stop words are excluded so generic phrases like "what is the capital of
    japan" don't accidentally match climate KB entries on words like
    'is', 'what', 'of', 'the'.  A minimum score of 2 is required so
    single-word coincidences (e.g. 'is' appearing everywhere) never
    produce a match.
    """
    # Common English/Filipino stop words that carry no topical meaning.
    STOP_WORDS = {
        "a","an","the","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","need","dare","ought",
        "of","in","on","at","to","for","with","by","from","as","into",
        "through","about","against","between","into","during","before",
        "after","above","below","up","down","out","off","over","under",
        "what","which","who","whom","whose","when","where","why","how",
        "this","that","these","those","it","its","they","them","their",
        "he","she","his","her","we","our","you","your","i","me","my",
        "and","or","but","if","because","so","yet","both","either",
        "not","no","nor","too","very","just","than","then","there",
        # Filipino particles
        "ang","ng","sa","na","ay","mga","at","si","ni","kay","nang",
        "kung","pa","din","rin","ba","po","nga","ito","iyan","iyon",
        "ako","ikaw","siya","kami","kayo","tayo","namin","nila","mo",
        "san","saan","tanghali","magandang","umaga","hapon"  # "where" in Filipino - very common question word
    }

    # Tokenise: lowercase words of >=2 chars, punctuation stripped.
    words = set(re.findall(r"[a-z0-9]+", query.lower())) - STOP_WORDS
    # Remove very short tokens after stop-word filtering
    words = {w for w in words if len(w) >= 3}

    # Nothing meaningful left → no KB match
    if not words:
        return []

    rows = []
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT id, category, question, answer_en, answer_tl, keywords "
            "FROM knowledge_base"
        )
        for r in cur.fetchall():
            haystack = " ".join(
                (r["question"] or "", r["keywords"] or "", r["answer_en"] or "")
            ).lower()
            score = sum(1 for w in words if len(w) >= 3 and w in haystack)
            # Accept score of 1 if query only has 1 significant content word
            min_score = 1 if len(words) == 1 else 2
            if score >= min_score:
                rows.append((score, r))
    finally:
        conn.close()

    rows.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "id": r["id"],
            "category": r["category"],
            "question": r["question"],
            "answer_en": r["answer_en"],
            "answer_tl": r["answer_tl"],
            "keywords": r["keywords"],
            "score": score,
        }
        for score, r in rows[:limit]
    ]


def add_knowledge(category, question, answer_en, answer_tl=None, keywords=None):
    """Insert a new knowledge_base entry and return its row id."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO knowledge_base (category, question, answer_en, answer_tl, keywords) "
            "VALUES (?, ?, ?, ?, ?)",
            (category, question, answer_en, answer_tl, keywords),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_chatlog(session_id, limit=None):
    """Return the message history for one session (chronological)."""
    conn = _get_conn()
    try:
        sql = (
            "SELECT id, session_id, role, message, lang, topic, created_at "
            "FROM chatlogs WHERE session_id = ? ORDER BY id ASC"
        )
        params = [session_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def chatlog_stats():
    """Quick counts for diagnostics / an admin view."""
    conn = _get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM chatlogs").fetchone()[0]
        sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM chatlogs"
        ).fetchone()[0]
        kb = conn.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]
        return {"messages": total, "sessions": sessions, "kb_entries": kb}
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Starter content. Covers topics the rule-based router does NOT already
# handle, so the knowledge-base fallback is genuinely useful out of the
# box. Re-running seed_knowledge_base() won't duplicate these.
# ----------------------------------------------------------------------
_STARTER_KB = [
    {
        "category": "agencies",
        "question": "What is PAGASA?",
        "answer_en": (
            "🌦️ **PAGASA** (Philippine Atmospheric, Geophysical and Astronomical "
            "Services Administration) is the Philippines' national weather "
            "agency. It monitors typhoons, issues El Niño/La Niña advisories, "
            "and releases seasonal rainfall forecasts that communities like "
            "Padre Garcia rely on for planning."
        ),
        "answer_tl": (
            "🌦️ Ang **PAGASA** (Philippine Atmospheric, Geophysical and "
            "Astronomical Services Administration) ang national weather "
            "agency ng Pilipinas. Sila ang nagmo-monitor ng bagyo, "
            "naglalabas ng El Niño/La Niña advisories, at seasonal rainfall "
            "forecasts na ginagamit ng mga komunidad tulad ng Padre Garcia "
            "para sa planning."
        ),
        "keywords": "pagasa,weather agency,dost,forecast,advisory,panahon,ahensya",
    },
    {
        "category": "agencies",
        "question": "What is MDRRMO?",
        "answer_en": (
            "🛡️ **MDRRMO** stands for Municipal Disaster Risk Reduction and "
            "Management Office. It is the local unit in a municipality (such "
            "as Padre Garcia) responsible for disaster preparedness, response, "
            "and early warning. For El Niño protocols specific to your area, "
            "coordinate with your MDRRMO."
        ),
        "answer_tl": (
            "🛡️ Ang **MDRRMO** (Municipal Disaster Risk Reduction and "
            "Management Office) ang local unit sa isang bayan (tulad ng "
            "Padre Garcia) na responsable sa disaster preparedness, "
            "response, at early warning. Para sa El Niño protocols na "
            "specific sa inyong lugar, makipag-ugnayan sa inyong MDRRMO."
        ),
        "keywords": "mdrrmo,disaster office,emergency,preparedness,local unit,sakuna",
    },
    {
        "category": "concepts",
        "question": "What is drought?",
        "answer_en": (
            "🏜️ **Drought** is a prolonged period of below-normal rainfall "
            "that causes water shortages. It is one of the main impacts of "
            "El Niño in the Philippines, reducing water for drinking, "
            "irrigation, and hydropower."
        ),
        "answer_tl": (
            "🏜️ Ang **drought** ay isang mahabang panahon ng below-average "
            "rainfall na nagdudulot ng water shortage. Ito ang pangunahing "
            "epekto ng El Niño sa Pilipinas — nababawasan ang water supply "
            "para sa inumin, irrigation, at hydropower."
        ),
        "keywords": "drought,tagtuyot,dry,water shortage,tuyo,kakulangan ng tubig",
    },
    {
        "category": "concepts",
        "question": "What is ENSO?",
        "answer_en": (
            "🔄 **ENSO** (El Niño–Southern Oscillation) is the natural "
            "climate cycle in the Pacific Ocean that swings between a warm "
            "phase (El Niño), a cool phase (La Niña), and a neutral phase. "
            "It is the driver behind the rainfall and temperature shifts "
            "ReadyKomunidad tracks across its three stations."
        ),
        "answer_tl": (
            "🔄 Ang **ENSO** (El Niño–Southern Oscillation) ang natural na "
            "climate cycle sa Pacific Ocean na nagpapalit-palit sa warm "
            "phase (El Niño), cool phase (La Niña), at neutral phase. Ito "
            "ang driver behind ng mga pagbabago sa rainfall at temperature "
            "na sinusubaybayan ng ReadyKomunidad sa tatlong stations."
        ),
        "keywords": "enso,el nino southern oscillation,cycle,pacific,driver,siklo",
    },
    {
        "category": "preparedness",
        "question": "How do I harvest rainwater?",
        "answer_en": (
            "🪣 **Rainwater harvesting** tips for El Niño preparedness:\n"
            "• Place clean containers under gutters and downspouts when rain is forecast\n"
            "• Cover stored water to keep out mosquitoes and debris\n"
            "• Use a first-flush diverter so the initial (dirtiest) runoff is discarded\n"
            "• Filter collected water (cloth/charcoal) before drinking, or boil it"
        ),
        "answer_tl": (
            "🪣 Mga tips sa **rainwater harvesting** para sa El Niño "
            "preparedness:\n"
            "• Ilagay ang malilinis na containers sa ilalim ng gutters at downspouts kapag may rain forecast\n"
            "• Takpan ang stored water para hindi pumasok ang lamok at debris\n"
            "• Gumamit ng first-flush diverter para itapon ang unang (pinakamaduming) runoff\n"
            "• Salain ang nakolektang water (cloth/charcoal) bago inumin, o pakuluin"
        ),
        "keywords": "rainwater,harvest,collect water,impok ng tubig,container,first flush,",
    },
]


def seed_knowledge_base():
    """Insert the starter entries only if the knowledge base is empty."""
    conn = _get_conn()
    try:
        count = conn.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]
        if count > 0:
            return
        conn.executemany(
            "INSERT INTO knowledge_base "
            "(category, question, answer_en, answer_tl, keywords) "
            "VALUES (:category, :question, :answer_en, :answer_tl, :keywords)",
            _STARTER_KB,
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database ready at: {DB_PATH}")
    print(chatlog_stats())
