import json
import uuid

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS usage_events (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_id  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    is_owner    BOOLEAN NOT NULL DEFAULT FALSE,
    detail      JSONB
)
"""
CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_usage_events_created ON usage_events"


@st.cache_resource(ttl=600)
def get_engine():
    """Connect to the database and create the table. Returns None if tracking is not set up or the database cannot be reached, so the app itself keeps working. Tries again every 10 minutes."""
    try:
        url = st.secrets["DATABASE_URL"]
    except Exception:
        return None
    try:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        engine = create_engine(url, pool_pre_ping=True, pool_recycle=300)
        with engine.begin() as conn:
            conn.execute(text(CREATE_TABLE))
            conn.execute(text(CREATE_INDEX))
        return engine
    except Exception:
        return None


def session_id():
    """A random id for this browser session. It contains no personal information."""
    if "sid" not in st.session_state:
        st.session_state["sid"] = uuid.uuid4().hex[:12]
    return st.session_state["sid"]


def is_owner():
    """Open the app once with ?owner=YOUR_KEY and your own visits are left out of the statistics."""
    try:
        key = st.secrets["OWNER_KEY"]
    except Exception:
        return False
    if st.query_params.get("owner") == key:
        st.session_state["owner"] = True
    return st.session_state.get("owner", False)


def log_event(event_type, detail=None):
    """Store one anonymous event. Never raises: tracking must not break the app."""
    engine = get_engine()
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO usage_events (session_id, event_type, is_owner, detail) "
                     "VALUES (:sid, :etype, :owner, CAST(:detail AS JSONB))"),
                {"sid": session_id(), "etype": event_type, "owner": is_owner(),
                 "detail": json.dumps(detail or {})})
    except Exception:
        pass


@st.cache_data(ttl=60)
def load_stats():
    """Numbers for the public usage dashboard (the owner's visits are excluded). Returns None if tracking is not available."""
    engine = get_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            overview = pd.read_sql(text("""
                SELECT
                    COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'visit')   AS visits,
                    COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'engaged') AS engaged,
                    COUNT(*) FILTER (WHERE event_type = 'example_selected')          AS example_clicks,
                    MIN(created_at)                                                  AS since
                FROM usage_events
                WHERE NOT is_owner"""), conn)
            daily = pd.read_sql(text("""
                SELECT CAST(created_at AS date) AS day,
                       COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'visit')   AS visits,
                       COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'engaged') AS engaged
                FROM usage_events
                WHERE NOT is_owner AND created_at > now() - interval '30 days'
                GROUP BY 1
                ORDER BY 1"""), conn)
            examples = pd.read_sql(text("""
                SELECT detail->>'example' AS example, COUNT(*) AS times
                FROM usage_events
                WHERE event_type = 'example_selected' AND NOT is_owner
                GROUP BY 1
                ORDER BY times DESC"""), conn)
            feedback = pd.read_sql(text("""
                SELECT COUNT(*) AS responses,
                       AVG(CAST(detail->>'rating' AS numeric)) AS avg_rating
                FROM usage_events
                WHERE event_type = 'feedback' AND NOT is_owner"""), conn)
        return {"overview": overview, "daily": daily, "examples": examples, "feedback": feedback}
    except Exception:
        return None