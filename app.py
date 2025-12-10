import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for older Python
    from backports.zoneinfo import ZoneInfo  # type: ignore

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
DB_PATH = os.getenv("DB_PATH", "db.sqlite3")
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "*")

# Разрешённые источники CORS
raw_origins = (ALLOWED_ORIGINS_RAW or "").strip()
if not raw_origins or raw_origins == "*":
    allow_origins = ["*"]
    allow_credentials = False  # wildcard не сочетается с credentials
else:
    allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    allow_credentials = True
    if not allow_origins:
        allow_origins = ["*"]
        allow_credentials = False

app = FastAPI(title="Count API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

TEN_DIGITS = re.compile(r"^\d{10}$")
KYIV_TZ = ZoneInfo("Europe/Kiev")
HOUR_MS = 60 * 60 * 1000


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                male_id TEXT NOT NULL,
                female_id TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                operator_name TEXT,
                man_name TEXT,
                man_age TEXT,
                man_city TEXT,
                woman_name TEXT,
                woman_age TEXT,
                woman_city TEXT,
                text TEXT,
                updated_at INTEGER NOT NULL,
                actions_total INTEGER DEFAULT 0,
                actions_paid INTEGER DEFAULT 0,
                balance_earned REAL DEFAULT 0,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(male_id, female_id, operator_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reports_female ON reports(female_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_male ON reports(male_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hourly_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                female_id TEXT NOT NULL,
                shift_key TEXT NOT NULL,
                hour_start INTEGER NOT NULL,
                operator_id TEXT NOT NULL,
                operator_name TEXT,
                actions_total INTEGER DEFAULT 0,
                actions_paid INTEGER DEFAULT 0,
                balance_earned REAL DEFAULT 0,
                chat_count INTEGER DEFAULT 0,
                mail_count INTEGER DEFAULT 0,
                paid_chat INTEGER DEFAULT 0,
                paid_mail INTEGER DEFAULT 0,
                updated_at INTEGER NOT NULL,
                UNIQUE(female_id, hour_start, operator_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hourly_stats_female_shift "
            "ON hourly_stats(female_id, shift_key)"
        )
    finally:
        conn.close()


init_db()


class PersonPayload(BaseModel):
    id: str = ""
    name: Optional[str] = None
    age: Optional[str] = None
    city: Optional[str] = None


class ReportPayload(BaseModel):
    male_id: str = Field(..., pattern=r"^\d{10}$")
    female_id: str = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1)
    operator_name: Optional[str] = None
    man: PersonPayload = PersonPayload()
    woman: PersonPayload = PersonPayload()
    text: str = ""
    updated_at: int = Field(..., ge=0)
    actions_total: int = Field(0, ge=0)
    actions_paid: int = Field(0, ge=0)
    balance_earned: float = 0.0


class HourlyStatPayload(BaseModel):
    female_id: str = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1)
    operator_name: Optional[str] = None
    hour_start: int = Field(..., ge=0)
    actions_total: int = Field(0, ge=0)
    actions_paid: int = Field(0, ge=0)
    balance_earned: float = 0.0
    chat_count: int = Field(0, ge=0)
    mail_count: int = Field(0, ge=0)
    paid_chat: int = Field(0, ge=0)
    paid_mail: int = Field(0, ge=0)

    @validator("hour_start")
    def validate_hour_start(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("hour_start must be positive")
        return value


class SyncPayload(BaseModel):
    reports: List[ReportPayload] = []
    hourly_stats: List[HourlyStatPayload] = []
    shift_key: Optional[str] = None


def auth(authorization: str | None = Header(default=None)):
    if not API_KEY:
        return True  # ключ отключен
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True


def compute_shift_key(ts_ms: int) -> str:
    if ts_ms <= 0:
        now = datetime.now(tz=KYIV_TZ) + timedelta(hours=1)
        return now.strftime("%Y-%m-%d")
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(KYIV_TZ)
    shifted = dt + timedelta(hours=1)
    return shifted.strftime("%Y-%m-%d")


def ensure_hour_start(ts_ms: int) -> int:
    return ts_ms - (ts_ms % HOUR_MS)


def get_count_from_db(male_id: str) -> int:
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="DB file not found")
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) AS c
            FROM messages m
            JOIN message_male_ids mm ON mm.message_id_ref = m.id
            WHERE mm.male_id = ?
            """,
            (male_id,),
        )
        row = cur.fetchone()
        conn.close()
        return int(row["c"] if row and row["c"] is not None else 0)
    except Exception as e:  # pragma: no cover - surface error for API response
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/count")
def count(male_id: str, _=Depends(auth)):
    if not TEN_DIGITS.match(male_id):
        raise HTTPException(status_code=400, detail="male_id must be exactly 10 digits")
    n = get_count_from_db(male_id)
    return {"ok": True, "male_id": male_id, "count": n}


def upsert_report(conn: sqlite3.Connection, payload: ReportPayload) -> bool:
    fields = {
        "male_id": payload.male_id,
        "female_id": payload.female_id.strip(),
        "operator_id": payload.operator_id.strip(),
        "operator_name": (payload.operator_name or "").strip() or None,
        "man_name": (payload.man.name or "").strip() or None,
        "man_age": (payload.man.age or "").strip() or None,
        "man_city": (payload.man.city or "").strip() or None,
        "woman_name": (payload.woman.name or "").strip() or None,
        "woman_age": (payload.woman.age or "").strip() or None,
        "woman_city": (payload.woman.city or "").strip() or None,
        "text": payload.text or "",
        "updated_at": int(payload.updated_at),
        "actions_total": int(payload.actions_total or 0),
        "actions_paid": int(payload.actions_paid or 0),
        "balance_earned": float(payload.balance_earned or 0),
    }
    cur = conn.execute(
        """
        INSERT INTO reports (
            male_id, female_id, operator_id, operator_name,
            man_name, man_age, man_city,
            woman_name, woman_age, woman_city,
            text, updated_at, actions_total, actions_paid, balance_earned
        )
        VALUES (
            :male_id, :female_id, :operator_id, :operator_name,
            :man_name, :man_age, :man_city,
            :woman_name, :woman_age, :woman_city,
            :text, :updated_at, :actions_total, :actions_paid, :balance_earned
        )
        ON CONFLICT(male_id, female_id, operator_id)
        DO UPDATE SET
            operator_name=excluded.operator_name,
            man_name=excluded.man_name,
            man_age=excluded.man_age,
            man_city=excluded.man_city,
            woman_name=excluded.woman_name,
            woman_age=excluded.woman_age,
            woman_city=excluded.woman_city,
            text=excluded.text,
            updated_at=excluded.updated_at,
            actions_total=excluded.actions_total,
            actions_paid=excluded.actions_paid,
            balance_earned=excluded.balance_earned
        WHERE excluded.updated_at >= reports.updated_at
        """,
        fields,
    )
    return cur.rowcount > 0


def upsert_hourly_stat(
    conn: sqlite3.Connection,
    payload: HourlyStatPayload,
    default_shift: Optional[str],
) -> bool:
    hour_start = ensure_hour_start(int(payload.hour_start))
    shift_key = default_shift or compute_shift_key(hour_start)
    params = {
        "female_id": payload.female_id.strip(),
        "shift_key": shift_key,
        "hour_start": hour_start,
        "operator_id": payload.operator_id.strip(),
        "operator_name": (payload.operator_name or "").strip() or None,
        "actions_total": int(payload.actions_total or 0),
        "actions_paid": int(payload.actions_paid or 0),
        "balance_earned": float(payload.balance_earned or 0),
        "chat_count": int(payload.chat_count or 0),
        "mail_count": int(payload.mail_count or 0),
        "paid_chat": int(payload.paid_chat or 0),
        "paid_mail": int(payload.paid_mail or 0),
        "updated_at": int(time.time() * 1000),
    }
    cur = conn.execute(
        """
        INSERT INTO hourly_stats (
            female_id, shift_key, hour_start, operator_id, operator_name,
            actions_total, actions_paid, balance_earned,
            chat_count, mail_count, paid_chat, paid_mail, updated_at
        ) VALUES (
            :female_id, :shift_key, :hour_start, :operator_id, :operator_name,
            :actions_total, :actions_paid, :balance_earned,
            :chat_count, :mail_count, :paid_chat, :paid_mail, :updated_at
        )
        ON CONFLICT(female_id, hour_start, operator_id)
        DO UPDATE SET
            operator_name=excluded.operator_name,
            shift_key=excluded.shift_key,
            actions_total=CASE
                WHEN excluded.actions_total > hourly_stats.actions_total
                THEN excluded.actions_total ELSE hourly_stats.actions_total END,
            actions_paid=CASE
                WHEN excluded.actions_paid > hourly_stats.actions_paid
                THEN excluded.actions_paid ELSE hourly_stats.actions_paid END,
            balance_earned=CASE
                WHEN excluded.balance_earned > hourly_stats.balance_earned
                THEN excluded.balance_earned ELSE hourly_stats.balance_earned END,
            chat_count=CASE
                WHEN excluded.chat_count > hourly_stats.chat_count
                THEN excluded.chat_count ELSE hourly_stats.chat_count END,
            mail_count=CASE
                WHEN excluded.mail_count > hourly_stats.mail_count
                THEN excluded.mail_count ELSE hourly_stats.mail_count END,
            paid_chat=CASE
                WHEN excluded.paid_chat > hourly_stats.paid_chat
                THEN excluded.paid_chat ELSE hourly_stats.paid_chat END,
            paid_mail=CASE
                WHEN excluded.paid_mail > hourly_stats.paid_mail
                THEN excluded.paid_mail ELSE hourly_stats.paid_mail END,
            updated_at=excluded.updated_at
        """,
        params,
    )
    return cur.rowcount > 0


@app.post("/api/reports/sync")
def sync_reports(payload: SyncPayload, _=Depends(auth)):
    if not payload.reports and not payload.hourly_stats:
        return {"ok": True, "updated_reports": 0, "updated_hourly": 0}
    conn = get_conn()
    try:
        updated_reports = 0
        updated_hourly = 0
        with conn:
            for report in payload.reports:
                if upsert_report(conn, report):
                    updated_reports += 1
            for stat in payload.hourly_stats:
                if upsert_hourly_stat(conn, stat, payload.shift_key):
                    updated_hourly += 1
        return {
            "ok": True,
            "updated_reports": updated_reports,
            "updated_hourly": updated_hourly,
        }
    finally:
        conn.close()


@app.get("/api/reports")
def list_reports(male_id: str, female_id: str, _=Depends(auth)):
    male_id = male_id.strip()
    female_id = female_id.strip()
    if not TEN_DIGITS.match(male_id):
        raise HTTPException(status_code=400, detail="male_id must be exactly 10 digits")
    if not female_id:
        raise HTTPException(status_code=400, detail="female_id is required")
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT *
            FROM reports
            WHERE male_id = ? AND female_id = ?
            ORDER BY updated_at DESC
            """,
            (male_id, female_id),
        )
        rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "items": rows}
    finally:
        conn.close()


@app.get("/api/history")
def history(female_id: str, shift_key: Optional[str] = None, _=Depends(auth)):
    female_id = female_id.strip()
    if not female_id:
        raise HTTPException(status_code=400, detail="female_id is required")
    key = shift_key or compute_shift_key(int(time.time() * 1000))
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT *
            FROM hourly_stats
            WHERE female_id = ? AND shift_key = ?
            ORDER BY hour_start DESC, operator_id ASC
            """,
            (female_id, key),
        )
        rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "items": rows, "shift_key": key}
    finally:
        conn.close()
