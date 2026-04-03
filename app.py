import io
import json
import logging
import math
import os
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for older Python
    from backports.zoneinfo import ZoneInfo  # type: ignore

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

logger = logging.getLogger("ot4et.count_api")

API_KEY = os.getenv("API_KEY", "")
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY", "") or "").strip()
ELEVENLABS_API_KEY = (os.getenv("ELEVENLABS_API_KEY", "") or "").strip()
ELEVENLABS_VOICE_1_ID = (os.getenv("ELEVENLABS_VOICE_1_ID", "") or "").strip()
ELEVENLABS_VOICE_1_LABEL = (os.getenv("ELEVENLABS_VOICE_1_LABEL", "") or "").strip()
ELEVENLABS_VOICE_2_ID = (os.getenv("ELEVENLABS_VOICE_2_ID", "") or "").strip()
ELEVENLABS_VOICE_2_LABEL = (os.getenv("ELEVENLABS_VOICE_2_LABEL", "") or "").strip()
ELEVENLABS_VOICE_3_ID = (os.getenv("ELEVENLABS_VOICE_3_ID", "") or "").strip()
ELEVENLABS_VOICE_3_LABEL = (os.getenv("ELEVENLABS_VOICE_3_LABEL", "") or "").strip()
ELEVENLABS_VOICE_SOFT_ID = (os.getenv("ELEVENLABS_VOICE_SOFT_ID", "") or "").strip()
ELEVENLABS_VOICE_FLIRTY_ID = (os.getenv("ELEVENLABS_VOICE_FLIRTY_ID", "") or "").strip()
ELEVENLABS_VOICE_CONFIDENT_ID = (os.getenv("ELEVENLABS_VOICE_CONFIDENT_ID", "") or "").strip()
ADMIN_TOKEN = (os.getenv("ADMIN_TOKEN", "") or "").strip()
DB_PATH = os.getenv("DB_PATH", "db.sqlite3")
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "*")
EXTENSION_ACCESS_PASSWORD = (os.getenv("EXTENSION_ACCESS_PASSWORD", "") or "").strip()
LATEST_EXTENSION_VERSION = (os.getenv("LATEST_EXTENSION_VERSION", "") or "").strip()

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
    expose_headers=[
        "X-Audio-Generation-Limit",
        "X-Audio-Generation-Used",
        "X-Audio-Generation-Remaining",
        "X-Audio-Filename",
        "Content-Disposition",
    ],
)

TEN_DIGITS = re.compile(r"^\d{10}$")
KYIV_TZ = ZoneInfo("Europe/Kiev")
HOUR_MS = 60 * 60 * 1000
REPORTS_SHIFT_RESET_HOUR = 23
OPERATOR_SHIFT_RESET_HOUR = 3
AUDIO_TRANSCRIBE_ALLOWED_HOSTS = {"chats-audios.cdndate.net"}
AUDIO_TRANSCRIBE_MAX_BYTES = 25 * 1024 * 1024
AUDIO_TRANSCRIBE_TIMEOUT_SECONDS = 20
ELEVENLABS_TTS_API_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_TTS_MODEL_ID = "eleven_flash_v2_5"
ELEVENLABS_TTS_OUTPUT_FORMAT = "mp3_44100_128"
ELEVENLABS_TTS_TIMEOUT_SECONDS = 30
ELEVENLABS_TTS_TEXT_MAX_LENGTH = 40000
AUDIO_GENERATION_LIMIT = 20
STATE_SECTIONS = {"reports", "hourly_stats", "chat_links", "history"}
GLOBAL_STATE_SECTIONS = {"top", "operator_names"}
GLOBAL_OPERATOR_NAMES_DAY_KEY = "global"

logger.info(
    "Loaded ElevenLabs env: api_key=%s voice_1=%s voice_2=%s voice_3=%s legacy_soft=%s legacy_flirty=%s legacy_confident=%s latest_extension=%s",
    bool(ELEVENLABS_API_KEY),
    ELEVENLABS_VOICE_1_ID or "<empty>",
    ELEVENLABS_VOICE_2_ID or "<empty>",
    ELEVENLABS_VOICE_3_ID or "<empty>",
    ELEVENLABS_VOICE_SOFT_ID or "<empty>",
    ELEVENLABS_VOICE_FLIRTY_ID or "<empty>",
    ELEVENLABS_VOICE_CONFIDENT_ID or "<empty>",
    LATEST_EXTENSION_VERSION or "<empty>",
)


class AudioTranscribeError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class AudioGenerateError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def get_conn() -> sqlite3.Connection:
    """Возвращает соединение с основной БД (reports + hourly_stats)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def create_reports_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            male_id TEXT NOT NULL,
            female_id TEXT NOT NULL,
            operator_id TEXT NOT NULL,
            operator_name TEXT,
            shift_key TEXT NOT NULL,
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
            UNIQUE(male_id, female_id, operator_id, shift_key)
        )
        """
    )


def create_reports_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reports_female ON reports(female_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reports_male ON reports(male_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reports_shift ON reports(shift_key, female_id)"
    )


def migrate_reports_table(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(reports)")
    columns = {row["name"] for row in cur.fetchall()}
    if "shift_key" in columns:
        create_reports_indexes(conn)
        return
    rows = conn.execute("SELECT * FROM reports").fetchall()
    with conn:
        conn.execute("ALTER TABLE reports RENAME TO reports_legacy")
        create_reports_table(conn)
        insert_sql = """
            INSERT INTO reports (
                male_id,
                female_id,
                operator_id,
                operator_name,
                shift_key,
                man_name,
                man_age,
                man_city,
                woman_name,
                woman_age,
                woman_city,
                text,
                updated_at,
                actions_total,
                actions_paid,
                balance_earned,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for row in rows:
            updated_at = int(row["updated_at"] or 0)
            if updated_at <= 0:
                updated_at = int(time.time() * 1000)
            shift_key = compute_shift_key(updated_at)
            conn.execute(
                insert_sql,
                (
                    row["male_id"],
                    row["female_id"],
                    row["operator_id"],
                    row["operator_name"],
                    shift_key,
                    row["man_name"],
                    row["man_age"],
                    row["man_city"],
                    row["woman_name"],
                    row["woman_age"],
                    row["woman_city"],
                    row["text"],
                    updated_at,
                    row["actions_total"],
                    row["actions_paid"],
                    row["balance_earned"],
                    row["created_at"],
                ),
            )
        conn.execute("DROP TABLE IF EXISTS reports_legacy")
    create_reports_indexes(conn)


def ensure_reports_table(conn: sqlite3.Connection) -> None:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reports'"
    )
    exists = cur.fetchone() is not None
    if not exists:
        create_reports_table(conn)
        create_reports_indexes(conn)
        return
    migrate_reports_table(conn)


def init_db():
    conn = get_conn()
    try:
        ensure_reports_table(conn)
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_id TEXT NOT NULL,
                day_key TEXT NOT NULL,
                section TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(operator_id, day_key, section)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operator_state_lookup "
            "ON operator_state(operator_id, day_key, section)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operators_top (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_id TEXT NOT NULL,
                day_key TEXT NOT NULL,
                operator_name TEXT,
                shift_balance REAL DEFAULT 0,
                hour_balance REAL DEFAULT 0,
                updated_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(operator_id, day_key)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operators_top_day_key "
            "ON operators_top(day_key)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operators_top_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_id TEXT NOT NULL UNIQUE,
                operator_name TEXT,
                record_balance REAL DEFAULT 0,
                updated_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operators_top_records_balance "
            "ON operators_top_records(record_balance DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operators_actions_top (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_id TEXT NOT NULL,
                day_key TEXT NOT NULL,
                operator_name TEXT,
                shift_actions INTEGER DEFAULT 0,
                hour_actions INTEGER DEFAULT 0,
                updated_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(operator_id, day_key)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operators_actions_top_day_key "
            "ON operators_actions_top(day_key)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operators_actions_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_id TEXT NOT NULL UNIQUE,
                operator_name TEXT,
                record_actions INTEGER DEFAULT 0,
                updated_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operators_actions_records_value "
            "ON operators_actions_records(record_actions DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_shift_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                day_key TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                operator_name TEXT,
                actions_total INTEGER DEFAULT 0,
                chat_count INTEGER DEFAULT 0,
                mail_count INTEGER DEFAULT 0,
                balance_earned REAL DEFAULT 0,
                updated_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(profile_id, day_key, operator_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_profile_shift_stats_day "
            "ON profile_shift_stats(day_key, profile_id)"
        )
        cur = conn.execute("PRAGMA table_info(profile_shift_stats)")
        columns = {row["name"] for row in cur.fetchall()}
        if "operator_name" not in columns:
            conn.execute("ALTER TABLE profile_shift_stats ADD COLUMN operator_name TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_shift_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_key TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                operator_name TEXT,
                balance_total REAL DEFAULT 0,
                last_balance_total REAL DEFAULT 0,
                last_balance_updated INTEGER DEFAULT 0,
                actions_total INTEGER DEFAULT 0,
                chat_count INTEGER DEFAULT 0,
                mail_count INTEGER DEFAULT 0,
                hour_actions_total INTEGER DEFAULT 0,
                hour_chat_count INTEGER DEFAULT 0,
                hour_mail_count INTEGER DEFAULT 0,
                hour_start INTEGER,
                updated_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(day_key, operator_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operator_shift_summary_day "
            "ON operator_shift_summary(day_key, operator_id)"
        )
        cur = conn.execute("PRAGMA table_info(operator_shift_summary)")
        columns = {row["name"] for row in cur.fetchall()}
        if "operator_name" not in columns:
            conn.execute("ALTER TABLE operator_shift_summary ADD COLUMN operator_name TEXT")
        if "last_balance_total" not in columns:
            conn.execute("ALTER TABLE operator_shift_summary ADD COLUMN last_balance_total REAL DEFAULT 0")
        if "last_balance_updated" not in columns:
            conn.execute("ALTER TABLE operator_shift_summary ADD COLUMN last_balance_updated INTEGER DEFAULT 0")
        if "hour_actions_total" not in columns:
            conn.execute("ALTER TABLE operator_shift_summary ADD COLUMN hour_actions_total INTEGER DEFAULT 0")
        if "hour_chat_count" not in columns:
            conn.execute("ALTER TABLE operator_shift_summary ADD COLUMN hour_chat_count INTEGER DEFAULT 0")
        if "hour_mail_count" not in columns:
            conn.execute("ALTER TABLE operator_shift_summary ADD COLUMN hour_mail_count INTEGER DEFAULT 0")
        if "hour_start" not in columns:
            conn.execute("ALTER TABLE operator_shift_summary ADD COLUMN hour_start INTEGER")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_hourly_balance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_key TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                hour_start INTEGER NOT NULL,
                balance_amount REAL DEFAULT 0,
                updated_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(day_key, operator_id, hour_start)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operator_hourly_balance_day "
            "ON operator_hourly_balance(day_key, operator_id, hour_start)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_hourly_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_key TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                hour_start INTEGER NOT NULL,
                chat_count INTEGER DEFAULT 0,
                mail_count INTEGER DEFAULT 0,
                actions_total INTEGER DEFAULT 0,
                updated_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(day_key, operator_id, hour_start)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operator_hourly_actions_day "
            "ON operator_hourly_actions(day_key, operator_id, hour_start)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_spend_max (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                male_id TEXT NOT NULL,
                female_id TEXT NOT NULL,
                max_spend_all_credits REAL NOT NULL DEFAULT 0,
                chat_uid TEXT,
                last_operator_id TEXT,
                last_operator_name TEXT,
                updated_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(male_id, female_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_spend_pair "
            "ON chat_spend_max(male_id, female_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_spend_male "
            "ON chat_spend_max(male_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_spend_updated "
            "ON chat_spend_max(updated_at DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_pair_selection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                male_id TEXT NOT NULL,
                female_id TEXT NOT NULL,
                voice_key TEXT NOT NULL,
                mood TEXT NOT NULL,
                operator_id TEXT,
                updated_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now') * 1000),
                UNIQUE(male_id, female_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audio_pair_selection_pair "
            "ON audio_pair_selection(male_id, female_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audio_pair_selection_updated "
            "ON audio_pair_selection(updated_at DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_operator_generation_usage (
                operator_id TEXT PRIMARY KEY,
                operator_name TEXT,
                agency_id TEXT,
                generation_count INTEGER NOT NULL DEFAULT 0,
                first_generated_at INTEGER,
                last_generated_at INTEGER,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audio_operator_generation_usage_agency "
            "ON audio_operator_generation_usage(agency_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audio_operator_generation_usage_updated "
            "ON audio_operator_generation_usage(updated_at DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extension_passwords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                password TEXT NOT NULL,
                team_name TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                deleted_at INTEGER,
                UNIQUE(name)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ext_pwd_active "
            "ON extension_passwords(is_active, deleted_at)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extension_password_usages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                password_id INTEGER NOT NULL,
                install_id TEXT NOT NULL,
                extension_version TEXT,
                first_used_at INTEGER NOT NULL,
                last_used_at INTEGER NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 1,
                UNIQUE(password_id, install_id),
                FOREIGN KEY(password_id) REFERENCES extension_passwords(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ext_usage_pwd "
            "ON extension_password_usages(password_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ext_usage_install "
            "ON extension_password_usages(install_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extension_password_usage_operators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                password_id INTEGER NOT NULL,
                install_id TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                agency_id TEXT,
                first_used_at INTEGER NOT NULL,
                last_used_at INTEGER NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 1,
                UNIQUE(password_id, install_id, operator_id),
                FOREIGN KEY(password_id) REFERENCES extension_passwords(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ext_usage_ops_pwd "
            "ON extension_password_usage_operators(password_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ext_usage_ops_operator "
            "ON extension_password_usage_operators(operator_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extension_install_registry (
                install_id TEXT PRIMARY KEY,
                first_seen_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL,
                created_with_version TEXT,
                last_seen_version TEXT,
                is_admin_install INTEGER NOT NULL DEFAULT 0,
                admin_reason TEXT,
                first_operator_id TEXT,
                current_operator_id TEXT,
                agency_id TEXT,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_install_registry_admin "
            "ON extension_install_registry(is_admin_install, last_seen_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_install_registry_operator "
            "ON extension_install_registry(current_operator_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_install_registry_agency "
            "ON extension_install_registry(agency_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS install_operator_history (
                install_id TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                agency_id TEXT,
                first_seen_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL,
                seen_count INTEGER NOT NULL DEFAULT 1,
                is_admin_context_seen INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (install_id, operator_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_install_operator_history_operator "
            "ON install_operator_history(operator_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_install_operator_history_agency "
            "ON install_operator_history(agency_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_install_binding (
                operator_id TEXT PRIMARY KEY,
                active_install_id TEXT NOT NULL,
                active_extension_version TEXT,
                agency_id TEXT,
                bound_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                previous_install_id TEXT,
                previous_replaced_at INTEGER
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operator_install_binding_install "
            "ON operator_install_binding(active_install_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operator_install_binding_agency "
            "ON operator_install_binding(agency_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_install_binding_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_id TEXT NOT NULL,
                install_id TEXT NOT NULL,
                extension_version TEXT,
                agency_id TEXT,
                bound_at INTEGER NOT NULL,
                replaced_at INTEGER,
                reason TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operator_install_binding_history_operator "
            "ON operator_install_binding_history(operator_id, bound_at DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_team_state (
                operator_id TEXT PRIMARY KEY,
                team_name TEXT NOT NULL,
                password_id INTEGER,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY(password_id) REFERENCES extension_passwords(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_operator_team_state_password "
            "ON operator_team_state(password_id)"
        )
        cur = conn.execute("PRAGMA table_info(extension_passwords)")
        columns = {row["name"] for row in cur.fetchall()}
        if "team_name" not in columns:
            conn.execute("ALTER TABLE extension_passwords ADD COLUMN team_name TEXT")
        cur = conn.execute("PRAGMA table_info(extension_password_usage_operators)")
        columns = {row["name"] for row in cur.fetchall()}
        if "agency_id" not in columns:
            conn.execute("ALTER TABLE extension_password_usage_operators ADD COLUMN agency_id TEXT")
        cur = conn.execute("PRAGMA table_info(extension_password_usages)")
        columns = {row["name"] for row in cur.fetchall()}
        if "extension_version" not in columns:
            conn.execute(
                "ALTER TABLE extension_password_usages ADD COLUMN extension_version TEXT"
            )
        recompute_operator_shift_action_totals_bulk(conn)
    finally:
        conn.close()


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
    shift_key: Optional[str] = None
    man: PersonPayload = PersonPayload()
    woman: PersonPayload = PersonPayload()
    text: str = ""
    updated_at: int = Field(..., ge=0)
    actions_total: int = Field(0, ge=0)
    actions_paid: int = Field(0, ge=0)
    balance_earned: float = 0.0

    @validator("shift_key", pre=True, always=True)
    def normalize_shift_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


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


class ReportShiftSnapshotPayload(BaseModel):
    male_id: str = Field(..., pattern=r"^\d{10}$")
    female_id: str = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1)
    operator_name: Optional[str] = None
    shift_key: Optional[str] = None
    text: str = ""
    updated_at: int = Field(..., ge=0)


class StateSectionPayload(BaseModel):
    updated_at: int = Field(..., ge=0)
    data: Any = None


class OperatorStatePayload(BaseModel):
    operator_id: str = Field(..., min_length=1)
    day_key: Optional[str] = None
    sections: Dict[str, StateSectionPayload] = Field(default_factory=dict)
    global_sections: Dict[str, StateSectionPayload] = Field(default_factory=dict)

    @validator("day_key", pre=True, always=True)
    def normalize_day_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None


class TopOperatorEntry(BaseModel):
    operator_id: str = Field(..., min_length=1)
    operator_name: Optional[str] = None
    shift_balance: float = 0.0
    hour_balance: float = 0.0
    updated_at: int = Field(..., ge=0)
    day_key: Optional[str] = None


class TopSyncPayload(BaseModel):
    operators: List[TopOperatorEntry] = []


class TopActionEntry(BaseModel):
    operator_id: str = Field(..., min_length=1)
    operator_name: Optional[str] = None
    shift_actions: int = 0
    hour_actions: int = 0
    updated_at: int = Field(..., ge=0)
    day_key: Optional[str] = None


class TopActionsSyncPayload(BaseModel):
    operators: List[TopActionEntry] = []


class ProfileShiftDeltaEntry(BaseModel):
    profile_id: str = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1)
    operator_name: Optional[str] = None
    actions_total: int = Field(0, ge=0)
    chat_count: int = Field(0, ge=0)
    mail_count: int = Field(0, ge=0)
    balance_earned: float = 0.0
    updated_at: int = Field(..., ge=0)


class ProfileShiftDeltaPayload(BaseModel):
    day_key: Optional[str] = None
    profiles: List[ProfileShiftDeltaEntry] = []


class ProfileShiftBatchPayload(BaseModel):
    day_key: Optional[str] = None
    profile_ids: List[str] = []


class OperatorShiftSnapshotPayload(BaseModel):
    day_key: Optional[str] = None
    operator_id: str = Field(..., min_length=1)
    operator_name: Optional[str] = None
    balance_total: float = 0.0
    actions_total: int = Field(0, ge=0)
    chat_count: int = Field(0, ge=0)
    mail_count: int = Field(0, ge=0)
    hour_actions_total: int = Field(0, ge=0)
    hour_chat_count: int = Field(0, ge=0)
    hour_mail_count: int = Field(0, ge=0)
    hour_start: Optional[int] = None
    updated_at: int = Field(..., ge=0)


class AudioTranscribePayload(BaseModel):
    audio_url: str = Field(..., min_length=1)


class AudioGeneratePayload(BaseModel):
    text: str = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1)
    operator_name: Optional[str] = None
    agency_id: Optional[str] = None
    voice_key: Optional[str] = None
    mood: Optional[str] = None
    voice_preset: Optional[str] = None

    @validator("text")
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        if len(normalized) > ELEVENLABS_TTS_TEXT_MAX_LENGTH:
            raise ValueError("text is too long")
        return normalized

    @validator("operator_id")
    def validate_operator_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("operator_id is required")
        if len(normalized) > 64:
            raise ValueError("operator_id is too long")
        return normalized

    @validator("operator_name")
    def validate_operator_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > 120:
            raise ValueError("operator_name is too long")
        return normalized

    @validator("agency_id")
    def validate_agency_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > 64:
            raise ValueError("agency_id is too long")
        return normalized

    @validator("voice_key")
    def validate_voice_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized not in {"voice_1", "voice_2", "voice_3"}:
            raise ValueError("voice_key is invalid")
        return normalized

    @validator("mood")
    def validate_mood(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized not in {"joyful", "normal", "sad"}:
            raise ValueError("mood is invalid")
        return normalized

    @validator("voice_preset")
    def validate_voice_preset(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized not in {"soft", "flirty", "confident"}:
            raise ValueError("voice_preset is invalid")
        return normalized


class AudioPairSelectionPayload(BaseModel):
    male_id: str = Field(..., pattern=r"^\d{10}$")
    female_id: str = Field(..., min_length=1)
    voice_key: str = Field(..., min_length=1)
    mood: str = Field(..., min_length=1)
    updated_at: int = Field(..., ge=0)
    operator_id: Optional[str] = None

    @validator("voice_key")
    def validate_voice_key(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"voice_1", "voice_2", "voice_3"}:
            raise ValueError("voice_key is invalid")
        return normalized

    @validator("mood")
    def validate_mood(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"joyful", "normal", "sad"}:
            raise ValueError("mood is invalid")
        return normalized

    @validator("operator_id")
    def validate_operator_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > 64:
            raise ValueError("operator_id is too long")
        return normalized


class ChatSpendUpsertPayload(BaseModel):
    male_id: str = Field(..., pattern=r"^\d{10}$")
    female_id: str = Field(..., min_length=1)
    max_spend_all_credits: float = Field(0.0, ge=0)
    chat_uid: Optional[str] = None
    operator_id: Optional[str] = None
    operator_name: Optional[str] = None
    updated_at: int = Field(..., ge=0)


class ExtensionAuthPayload(BaseModel):
    password: str = Field(..., min_length=1)
    install_id: Optional[str] = None
    operator_id: Optional[str] = None
    agency_id: Optional[str] = None
    count_success: bool = True
    is_admin_context: bool = False

    @validator("install_id")
    def validate_install_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > 128:
            raise ValueError("install_id is too long")
        return normalized

    @validator("operator_id")
    def validate_operator_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > 64:
            raise ValueError("operator_id is too long")
        return normalized

    @validator("agency_id")
    def validate_agency_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > 64:
            raise ValueError("agency_id is too long")
        return normalized


class AdminPasswordCreatePayload(BaseModel):
    name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    team_name: Optional[str] = None
    is_active: bool = True

    @validator("team_name")
    def validate_team_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > 120:
            raise ValueError("team_name is too long")
        return normalized


class AdminPasswordUpdatePayload(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = None
    team_name: Optional[str] = None
    is_active: Optional[bool] = None

    @validator("name")
    def validate_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("name must not be empty")
        return normalized

    @validator("password")
    def validate_password(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("password must not be empty")
        return normalized

    @validator("team_name")
    def validate_update_team_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return ""
        if len(normalized) > 120:
            raise ValueError("team_name is too long")
        return normalized


def require_latest_extension_version(
    x_extension_version: str | None = Header(default=None),
):
    required = LATEST_EXTENSION_VERSION.strip()
    if not required:
        return True
    provided = (x_extension_version or "").strip()
    if provided != required:
        raise HTTPException(
            status_code=426,
            detail={
                "error": "extension_update_required",
                "required_version": required,
                "provided_version": provided,
            },
        )
    return True


def auth(
    authorization: str | None = Header(default=None),
    _=Depends(require_latest_extension_version),
):
    if not API_KEY:
        return True  # ключ отключен
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True


def admin_auth(x_admin_token: str | None = Header(default=None)):
    if not x_admin_token:
        raise HTTPException(status_code=401, detail="Admin token is required")
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin token is not configured")
    if x_admin_token.strip() != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True


def normalize_audio_transcribe_url(audio_url: str) -> str:
    raw = (audio_url or "").strip()
    if not raw:
        raise AudioTranscribeError("bad_url")
    try:
        parsed = urlparse(raw)
    except Exception as exc:
        raise AudioTranscribeError("bad_url") from exc
    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").lower()
    if scheme != "https":
        raise AudioTranscribeError("bad_url")
    if hostname not in AUDIO_TRANSCRIBE_ALLOWED_HOSTS:
        raise AudioTranscribeError("bad_host")
    path = parsed.path or ""
    if not path.lower().endswith(".mp3"):
        raise AudioTranscribeError("bad_url")
    return raw


def download_audio_transcribe_bytes(audio_url: str) -> bytes:
    req = Request(
        audio_url,
        headers={
            "User-Agent": "OT4ET/1.0",
            "Accept": "audio/mpeg,audio/*;q=0.9,*/*;q=0.1",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=AUDIO_TRANSCRIBE_TIMEOUT_SECONDS) as response:
            content_length_raw = response.headers.get("Content-Length", "").strip()
            if content_length_raw:
                try:
                    content_length = int(content_length_raw)
                except ValueError:
                    content_length = 0
                if content_length > AUDIO_TRANSCRIBE_MAX_BYTES:
                    raise AudioTranscribeError("file_too_large")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > AUDIO_TRANSCRIBE_MAX_BYTES:
                    raise AudioTranscribeError("file_too_large")
                chunks.append(chunk)
            audio_bytes = b"".join(chunks)
            if not audio_bytes:
                raise AudioTranscribeError("fetch_failed")
            return audio_bytes
    except AudioTranscribeError:
        raise
    except Exception as exc:
        raise AudioTranscribeError("fetch_failed") from exc


def transcribe_audio_bytes(audio_url: str, audio_bytes: bytes) -> str:
    if not OPENAI_API_KEY:
        raise AudioTranscribeError("openai_not_configured")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise AudioTranscribeError("openai_dependency_missing") from exc
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=AUDIO_TRANSCRIBE_TIMEOUT_SECONDS)
        parsed = urlparse(audio_url)
        filename = os.path.basename(parsed.path or "") or "audio.mp3"
        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = filename
        response = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=file_obj,
        )
        transcript = ""
        if isinstance(response, dict):
            transcript = str(response.get("text") or "").strip()
        else:
            transcript = str(getattr(response, "text", "") or "").strip()
        if not transcript:
            raise AudioTranscribeError("transcription_failed")
        return transcript
    except AudioTranscribeError:
        raise
    except Exception as exc:
        raise AudioTranscribeError("transcription_failed") from exc


def get_elevenlabs_voice_slots() -> Dict[str, Dict[str, Any]]:
    return {
        "voice_1": {
            "voice_id": str(ELEVENLABS_VOICE_1_ID or ELEVENLABS_VOICE_SOFT_ID or "").strip(),
            "label": str(ELEVENLABS_VOICE_1_LABEL or "").strip() or "Voice 1",
        },
        "voice_2": {
            "voice_id": str(ELEVENLABS_VOICE_2_ID or ELEVENLABS_VOICE_FLIRTY_ID or "").strip(),
            "label": str(ELEVENLABS_VOICE_2_LABEL or "").strip() or "Voice 2",
        },
        "voice_3": {
            "voice_id": str(ELEVENLABS_VOICE_3_ID or ELEVENLABS_VOICE_CONFIDENT_ID or "").strip(),
            "label": str(ELEVENLABS_VOICE_3_LABEL or "").strip() or "Voice 3",
        },
    }


def get_elevenlabs_voice_list() -> list[Dict[str, Any]]:
    voices: list[Dict[str, Any]] = []
    for key, config in get_elevenlabs_voice_slots().items():
        voices.append(
            {
                "key": key,
                "label": config["label"],
                "configured": bool(str(config.get("voice_id") or "").strip()),
            }
        )
    return voices


def get_elevenlabs_mood_profiles() -> Dict[str, Dict[str, Any]]:
    return {
        "joyful": {
            "stability": 0.58,
            "similarity_boost": 0.88,
            "style": 0.34,
            "speed": 0.82,
            "use_speaker_boost": True,
        },
        "normal": {
            "stability": 0.78,
            "similarity_boost": 0.86,
            "style": 0.14,
            "speed": 0.76,
            "use_speaker_boost": True,
        },
        "sad": {
            "stability": 0.72,
            "similarity_boost": 0.9,
            "style": 0.06,
            "speed": 0.7,
            "use_speaker_boost": True,
        },
    }


def resolve_elevenlabs_generate_selection(payload: AudioGeneratePayload) -> tuple[str, str]:
    voice_key = str(payload.voice_key or "").strip().lower()
    mood = str(payload.mood or "").strip().lower()
    if voice_key:
        if not mood:
            raise AudioGenerateError("invalid_mood")
        return voice_key, mood
    legacy_preset = str(payload.voice_preset or "").strip().lower()
    legacy_map = {
        "soft": "voice_1",
        "flirty": "voice_2",
        "confident": "voice_3",
    }
    if not legacy_preset:
        raise AudioGenerateError("invalid_voice_key")
    voice_key = legacy_map.get(legacy_preset)
    if not voice_key:
        raise AudioGenerateError("invalid_preset")
    return voice_key, "normal"


def get_elevenlabs_voice_config(voice_key: str, mood: str) -> Dict[str, Any]:
    normalized_voice_key = str(voice_key or "").strip().lower()
    normalized_mood = str(mood or "").strip().lower()
    voice_config = get_elevenlabs_voice_slots().get(normalized_voice_key)
    if not voice_config:
        raise AudioGenerateError("invalid_voice_key")
    mood_profile = get_elevenlabs_mood_profiles().get(normalized_mood)
    if not mood_profile:
        raise AudioGenerateError("invalid_mood")
    voice_id = str(voice_config.get("voice_id") or "").strip()
    if not voice_id:
        raise AudioGenerateError("voice_not_configured")
    return {
        "voice_key": normalized_voice_key,
        "mood": normalized_mood,
        "voice_id": voice_id,
        "voice_settings": dict(mood_profile),
        "label": str(voice_config.get("label") or "").strip() or normalized_voice_key,
    }


def upsert_audio_pair_selection(
    conn: sqlite3.Connection, payload: AudioPairSelectionPayload
) -> bool:
    params = {
        "male_id": payload.male_id.strip(),
        "female_id": payload.female_id.strip(),
        "voice_key": payload.voice_key.strip().lower(),
        "mood": payload.mood.strip().lower(),
        "operator_id": (payload.operator_id or "").strip() or None,
        "updated_at": int(payload.updated_at or 0),
    }
    row = conn.execute(
        """
        SELECT voice_key, mood, operator_id, updated_at
        FROM audio_pair_selection
        WHERE male_id = ? AND female_id = ?
        """,
        (params["male_id"], params["female_id"]),
    ).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO audio_pair_selection (
                male_id, female_id, voice_key, mood, operator_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                params["male_id"],
                params["female_id"],
                params["voice_key"],
                params["mood"],
                params["operator_id"],
                params["updated_at"],
            ),
        )
        return True
    changed = (
        str(row["voice_key"] or "").strip() != params["voice_key"]
        or str(row["mood"] or "").strip() != params["mood"]
        or str(row["operator_id"] or "").strip() != str(params["operator_id"] or "").strip()
        or int(row["updated_at"] or 0) != params["updated_at"]
    )
    if not changed:
        return False
    conn.execute(
        """
        UPDATE audio_pair_selection
        SET voice_key = ?, mood = ?, operator_id = ?, updated_at = ?
        WHERE male_id = ? AND female_id = ?
        """,
        (
            params["voice_key"],
            params["mood"],
            params["operator_id"],
            params["updated_at"],
            params["male_id"],
            params["female_id"],
        ),
    )
    return True


def fetch_audio_pair_selection(
    conn: sqlite3.Connection, male_id: str, female_id: str
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT male_id, female_id, voice_key, mood, operator_id, updated_at
        FROM audio_pair_selection
        WHERE male_id = ? AND female_id = ?
        """,
        (str(male_id or "").strip(), str(female_id or "").strip()),
    ).fetchone()
    if row is None:
        return None
    return {
        "male_id": row["male_id"],
        "female_id": row["female_id"],
        "voice_key": row["voice_key"],
        "mood": row["mood"],
        "operator_id": row["operator_id"],
        "updated_at": int(row["updated_at"] or 0),
    }


def get_audio_generation_usage(
    conn: sqlite3.Connection, operator_id: str
) -> Optional[Dict[str, Any]]:
    normalized_operator_id = str(operator_id or "").strip()
    if not normalized_operator_id:
        return None
    row = conn.execute(
        """
        SELECT
            operator_id,
            operator_name,
            agency_id,
            generation_count,
            first_generated_at,
            last_generated_at,
            updated_at
        FROM audio_operator_generation_usage
        WHERE operator_id = ?
        """,
        (normalized_operator_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "operator_id": str(row["operator_id"] or "").strip(),
        "operator_name": str(row["operator_name"] or "").strip(),
        "agency_id": str(row["agency_id"] or "").strip(),
        "generation_count": int(row["generation_count"] or 0),
        "first_generated_at": int(row["first_generated_at"] or 0),
        "last_generated_at": int(row["last_generated_at"] or 0),
        "updated_at": int(row["updated_at"] or 0),
    }


def build_audio_generation_quota_payload(
    conn: sqlite3.Connection, operator_id: str
) -> Dict[str, Any]:
    normalized_operator_id = str(operator_id or "").strip()
    usage = get_audio_generation_usage(conn, normalized_operator_id)
    used = int(usage["generation_count"] if usage else 0)
    remaining = max(0, AUDIO_GENERATION_LIMIT - used)
    return {
        "operator_id": normalized_operator_id,
        "limit": AUDIO_GENERATION_LIMIT,
        "used": used,
        "remaining": remaining,
        "blocked": remaining <= 0,
    }


def increment_audio_generation_usage(
    conn: sqlite3.Connection,
    operator_id: str,
    operator_name: Optional[str],
    agency_id: Optional[str],
    now_ms: int,
) -> Dict[str, Any]:
    normalized_operator_id = str(operator_id or "").strip()
    if not normalized_operator_id:
        raise ValueError("operator_id is required")
    normalized_operator_name = str(operator_name or "").strip() or None
    normalized_agency_id = str(agency_id or "").strip() or None
    row = conn.execute(
        """
        SELECT
            operator_id,
            operator_name,
            agency_id,
            generation_count,
            first_generated_at,
            last_generated_at
        FROM audio_operator_generation_usage
        WHERE operator_id = ?
        """,
        (normalized_operator_id,),
    ).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO audio_operator_generation_usage (
                operator_id,
                operator_name,
                agency_id,
                generation_count,
                first_generated_at,
                last_generated_at,
                updated_at
            ) VALUES (?, ?, ?, 1, ?, ?, ?)
            """,
            (
                normalized_operator_id,
                normalized_operator_name,
                normalized_agency_id,
                now_ms,
                now_ms,
                now_ms,
            ),
        )
    else:
        next_count = int(row["generation_count"] or 0) + 1
        next_operator_name = normalized_operator_name or str(row["operator_name"] or "").strip() or None
        next_agency_id = normalized_agency_id or str(row["agency_id"] or "").strip() or None
        first_generated_at = int(row["first_generated_at"] or 0) or now_ms
        conn.execute(
            """
            UPDATE audio_operator_generation_usage
            SET
                operator_name = ?,
                agency_id = ?,
                generation_count = ?,
                first_generated_at = ?,
                last_generated_at = ?,
                updated_at = ?
            WHERE operator_id = ?
            """,
            (
                next_operator_name,
                next_agency_id,
                next_count,
                first_generated_at,
                now_ms,
                now_ms,
                normalized_operator_id,
            ),
        )
    return build_audio_generation_quota_payload(conn, normalized_operator_id)


def build_elevenlabs_audio_filename(voice_key: str, mood: str) -> str:
    safe_voice_key = str(voice_key or "").strip().lower() or "voice"
    safe_mood = str(mood or "").strip().lower() or "normal"
    return f"ot4et-{safe_voice_key}-{safe_mood}-{int(time.time() * 1000)}.mp3"


def map_elevenlabs_http_error(exc: HTTPError) -> str:
    status = int(getattr(exc, "code", 0) or 0)
    body = ""
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    logger.warning("ElevenLabs HTTP %s response: %s", status, body[:500] if body else "<empty>")
    payload = body.lower()
    if status == 401 and "invalid_api_key" in payload:
        return "elevenlabs_invalid_api_key"
    if status in {401, 403} and (
        "missing_permissions" in payload
        or "needs_authorization" in payload
        or "authorization" in payload
    ):
        return "elevenlabs_permission_denied"
    if status == 400 and (
        "voice_settings" in payload
        or "speed" in payload
        or "stability" in payload
        or "similarity_boost" in payload
        or "style" in payload
        or "speaker_boost" in payload
        or "validation" in payload
        or "request body" in payload
    ):
        return "elevenlabs_invalid_voice_settings"
    if status in {400, 404, 422} and (
        "voice_not_found" in payload
        or "voice not found" in payload
        or "voice does not exist" in payload
        or "voice unavailable" in payload
        or "voice is unavailable" in payload
        or "unavailable voice" in payload
        or "unknown voice" in payload
    ):
        return "elevenlabs_voice_unavailable"
    return "generation_failed"


def generate_elevenlabs_audio_bytes(text: str, voice_key: str, mood: str) -> tuple[bytes, str]:
    if not ELEVENLABS_API_KEY:
        raise AudioGenerateError("elevenlabs_not_configured")
    config = get_elevenlabs_voice_config(voice_key, mood)
    voice_id = config["voice_id"]
    url = (
        f"{ELEVENLABS_TTS_API_BASE}/{voice_id}"
        f"?output_format={ELEVENLABS_TTS_OUTPUT_FORMAT}"
    )
    payload = {
        "text": str(text or "").strip(),
        "model_id": ELEVENLABS_TTS_MODEL_ID,
        "voice_settings": config["voice_settings"],
    }
    req = Request(
        url,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    try:
        with urlopen(req, timeout=ELEVENLABS_TTS_TIMEOUT_SECONDS) as response:
            audio_bytes = response.read()
            if not audio_bytes:
                raise AudioGenerateError("generation_failed")
            return audio_bytes, build_elevenlabs_audio_filename(
                config["voice_key"],
                config["mood"],
            )
    except AudioGenerateError:
        raise
    except HTTPError as exc:
        raise AudioGenerateError(map_elevenlabs_http_error(exc)) from exc
    except URLError as exc:
        raise AudioGenerateError("generation_failed") from exc
    except Exception as exc:
        raise AudioGenerateError("generation_failed") from exc


def compute_report_shift_key(ts_ms: int) -> str:
    if ts_ms <= 0:
        now = datetime.now(tz=KYIV_TZ)
        if now.hour >= REPORTS_SHIFT_RESET_HOUR:
            now = now + timedelta(days=1)
        return now.strftime("%Y-%m-%d")
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(KYIV_TZ)
    if dt.hour >= REPORTS_SHIFT_RESET_HOUR:
        dt = dt + timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def compute_operator_shift_day_key(ts_ms: int) -> str:
    if ts_ms <= 0:
        now = datetime.now(tz=KYIV_TZ)
        if now.hour < OPERATOR_SHIFT_RESET_HOUR:
            now = now - timedelta(days=1)
        return now.strftime("%Y-%m-%d")
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(KYIV_TZ)
    if dt.hour < OPERATOR_SHIFT_RESET_HOUR:
        dt = dt - timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def get_report_day_range(day_key: Optional[str]) -> Tuple[int, int]:
    """
    Возвращает интервал [start_ms, end_ms) для указанного report day_key
    в часовом поясе Киева, где день начинается в 23:00 предыдущего дня.
    """
    if day_key:
        try:
            y, m, d = [int(part) for part in day_key.split("-")]
            end_local = datetime(y, m, d, REPORTS_SHIFT_RESET_HOUR, 0, 0, tzinfo=KYIV_TZ)
        except Exception:
            end_local = datetime.now(tz=KYIV_TZ)
    else:
        normalized_day_key = compute_report_shift_key(int(time.time() * 1000))
        y, m, d = [int(part) for part in normalized_day_key.split("-")]
        end_local = datetime(y, m, d, REPORTS_SHIFT_RESET_HOUR, 0, 0, tzinfo=KYIV_TZ)
    start_local = end_local - timedelta(days=1)
    start_ms = int(start_local.timestamp() * 1000)
    end_ms = int(end_local.timestamp() * 1000)
    return start_ms, end_ms


def get_operator_shift_day_range(day_key: Optional[str]) -> Tuple[int, int]:
    """
    Возвращает интервал [start_ms, end_ms) для operator shift day_key
    в часовом поясе Киева, где день начинается в 03:00.
    """
    if day_key:
        try:
            y, m, d = [int(part) for part in day_key.split("-")]
            start_local = datetime(y, m, d, OPERATOR_SHIFT_RESET_HOUR, 0, 0, tzinfo=KYIV_TZ)
        except Exception:
            start_local = datetime.now(tz=KYIV_TZ)
    else:
        now_local = datetime.now(tz=KYIV_TZ)
        if now_local.hour < OPERATOR_SHIFT_RESET_HOUR:
            base = now_local - timedelta(days=1)
        else:
            base = now_local
        start_local = datetime(
            base.year,
            base.month,
            base.day,
            OPERATOR_SHIFT_RESET_HOUR,
            0,
            0,
            tzinfo=KYIV_TZ,
        )
    start_ms = int(start_local.timestamp() * 1000)
    end_ms = int((start_local + timedelta(days=1)).timestamp() * 1000)
    return start_ms, end_ms


def compute_shift_key(ts_ms: int) -> str:
    return compute_report_shift_key(ts_ms)


def get_kyiv_day_range(day_key: Optional[str]) -> Tuple[int, int]:
    return get_report_day_range(day_key)


def ensure_hour_start(ts_ms: int) -> int:
    return ts_ms - (ts_ms % HOUR_MS)


def normalize_state_day_key(raw: Optional[str]) -> str:
    value = (raw or "").strip()
    if value:
        return value
    return compute_report_shift_key(int(time.time() * 1000))


def normalize_operator_shift_day_key(raw: Optional[str]) -> str:
    value = (raw or "").strip()
    if value:
        return value
    return compute_operator_shift_day_key(int(time.time() * 1000))


def normalize_top_day_key(raw: Optional[str]) -> str:
    """
    Нормализует day_key для таблицы operators_top.
    Если значение пустое – используем текущее смещение смены по Киеву.
    """
    return normalize_state_day_key(raw)


def recompute_operator_shift_action_totals(
    conn: sqlite3.Connection, day_key: str, operator_id: str
) -> bool:
    normalized_day_key = normalize_operator_shift_day_key(day_key)
    normalized_operator_id = str(operator_id or "").strip()
    if not normalized_day_key or not normalized_operator_id:
        return False
    current = conn.execute(
        """
        SELECT actions_total, chat_count, mail_count, updated_at
        FROM operator_shift_summary
        WHERE day_key = ? AND operator_id = ?
        """,
        (normalized_day_key, normalized_operator_id),
    ).fetchone()
    if not current:
        return False
    aggregated = conn.execute(
        """
        SELECT
            COUNT(*) AS row_count,
            COALESCE(SUM(actions_total), 0) AS actions_total,
            COALESCE(SUM(chat_count), 0) AS chat_count,
            COALESCE(SUM(mail_count), 0) AS mail_count,
            MAX(updated_at) AS updated_at
        FROM operator_hourly_actions
        WHERE day_key = ? AND operator_id = ?
        """,
        (normalized_day_key, normalized_operator_id),
    ).fetchone()
    if not aggregated or int(aggregated["row_count"] or 0) <= 0:
        return False
    next_actions = int(aggregated["actions_total"] or 0)
    next_chat = int(aggregated["chat_count"] or 0)
    next_mail = int(aggregated["mail_count"] or 0)
    current_updated = int(current["updated_at"] or 0)
    next_updated = max(current_updated, int(aggregated["updated_at"] or 0))
    if (
        int(current["actions_total"] or 0) == next_actions
        and int(current["chat_count"] or 0) == next_chat
        and int(current["mail_count"] or 0) == next_mail
        and current_updated == next_updated
    ):
        return False
    conn.execute(
        """
        UPDATE operator_shift_summary
        SET actions_total = ?,
            chat_count = ?,
            mail_count = ?,
            updated_at = ?
        WHERE day_key = ? AND operator_id = ?
        """,
        (
            next_actions,
            next_chat,
            next_mail,
            next_updated,
            normalized_day_key,
            normalized_operator_id,
        ),
    )
    return True


def recompute_operator_shift_action_totals_bulk(
    conn: sqlite3.Connection, day_key: Optional[str] = None
) -> int:
    if day_key:
        normalized_day_key = normalize_top_day_key(day_key)
        rows = conn.execute(
            """
            SELECT DISTINCT s.day_key, s.operator_id
            FROM operator_shift_summary s
            INNER JOIN operator_hourly_actions h
                ON h.day_key = s.day_key AND h.operator_id = s.operator_id
            WHERE s.day_key = ?
            ORDER BY s.day_key ASC, s.operator_id ASC
            """,
            (normalized_day_key,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT DISTINCT s.day_key, s.operator_id
            FROM operator_shift_summary s
            INNER JOIN operator_hourly_actions h
                ON h.day_key = s.day_key AND h.operator_id = s.operator_id
            ORDER BY s.day_key ASC, s.operator_id ASC
            """
        ).fetchall()
    changed = 0
    for row in rows:
        if recompute_operator_shift_action_totals(
            conn,
            str(row["day_key"] or "").strip(),
            str(row["operator_id"] or "").strip(),
        ):
            changed += 1
    return changed


init_db()


def serialize_section_payload(data: Any) -> str:
    try:
        return json.dumps(data if data is not None else {}, ensure_ascii=False)
    except TypeError:
        return json.dumps({}, ensure_ascii=False)


def upsert_state_section(
    conn: sqlite3.Connection,
    operator_id: str,
    day_key: str,
    section: str,
    updated_at: int,
    data: Any,
) -> bool:
    section_key = section.strip()
    section_set = STATE_SECTIONS | GLOBAL_STATE_SECTIONS
    if section_key not in section_set:
        return False
    ts = int(updated_at or 0)
    if ts <= 0:
        ts = int(time.time() * 1000)
    target_operator = operator_id
    target_day_key = day_key
    payload_json = serialize_section_payload(data)
    existing_payload: Any = None
    existing_updated = 0
    cur = conn.execute(
        """
        SELECT updated_at, payload
        FROM operator_state
        WHERE operator_id = ? AND day_key = ? AND section = ?
        """,
        (target_operator, target_day_key, section_key),
    )
    row = cur.fetchone()
    if row:
        existing_updated = int(row["updated_at"] or 0)
        raw_payload = row["payload"]
        if isinstance(raw_payload, str) and raw_payload.strip():
            try:
                existing_payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                existing_payload = None
    changed = False
    final_payload = data
    if section_key in GLOBAL_STATE_SECTIONS:
        target_operator = "__GLOBAL__"
        if section_key == "operator_names":
            target_day_key = GLOBAL_OPERATOR_NAMES_DAY_KEY
        existing_list = (
            existing_payload if isinstance(existing_payload, list) else []
        )
        incoming_list = data if isinstance(data, list) else []
        if section_key == "top":
            final_payload, merged_changed = merge_global_top_entries(
                existing_list, incoming_list, day_key
            )
        else:
            final_payload, merged_changed = merge_global_operator_names(
                existing_list, incoming_list
            )
        if not merged_changed:
            return False
        payload_json = serialize_section_payload(final_payload)
        max_ts = max(
            (
                int(item.get("updated_at") or item.get("updatedAt") or 0)
                for item in final_payload
            ),
            default=ts,
        )
        ts = max(ts, max_ts)
        changed = True
    else:
        if existing_updated >= ts:
            return False
        changed = True
    if not changed:
        return False
    conn.execute(
        """
        INSERT INTO operator_state (
            operator_id, day_key, section, updated_at, payload
        ) VALUES (
            ?, ?, ?, ?, ?
        )
        ON CONFLICT(operator_id, day_key, section)
        DO UPDATE SET
            updated_at=excluded.updated_at,
            payload=excluded.payload
        """,
        (target_operator, target_day_key, section_key, ts, payload_json),
    )
    return True


def upsert_top_entry(conn: sqlite3.Connection, entry: TopOperatorEntry) -> bool:
    operator_id = entry.operator_id.strip()
    if not operator_id:
        return False
    day_key = normalize_top_day_key(entry.day_key)
    shift_balance = float(entry.shift_balance or 0.0)
    hour_balance = float(entry.hour_balance or 0.0)
    updated_at = int(entry.updated_at or 0) or int(time.time() * 1000)
    operator_name = (entry.operator_name or "").strip() or None

    cur = conn.execute(
        """
        SELECT operator_name, shift_balance, hour_balance, updated_at
        FROM operators_top
        WHERE operator_id = ? AND day_key = ?
        """,
        (operator_id, day_key),
    )
    row = cur.fetchone()
    if not row:
        conn.execute(
            """
            INSERT INTO operators_top (
                operator_id, day_key, operator_name,
                shift_balance, hour_balance, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (operator_id, day_key, operator_name, shift_balance, hour_balance, updated_at),
        )
        return True

    current_shift = float(row["shift_balance"] or 0.0)
    current_hour = float(row["hour_balance"] or 0.0)
    current_updated = int(row["updated_at"] or 0)
    current_name = (row["operator_name"] or "").strip() or None

    next_shift = current_shift
    next_hour = current_hour
    changed = False

    if shift_balance > current_shift:
        next_shift = shift_balance
        changed = True
    if hour_balance > current_hour:
        next_hour = hour_balance
        changed = True

    next_name = current_name
    if operator_name and operator_name != current_name:
        next_name = operator_name
        changed = True

    if not changed:
        return False

    next_updated = max(current_updated, updated_at or int(time.time() * 1000))
    conn.execute(
        """
        UPDATE operators_top
        SET operator_name = ?,
            shift_balance = ?,
            hour_balance = ?,
            updated_at = ?
        WHERE operator_id = ? AND day_key = ?
        """,
        (next_name, next_shift, next_hour, next_updated, operator_id, day_key),
    )
    return True


def upsert_top_record_entry(conn: sqlite3.Connection, entry: TopOperatorEntry) -> bool:
    operator_id = entry.operator_id.strip()
    if not operator_id:
        return False
    shift_balance = float(entry.shift_balance or 0.0)
    updated_at = int(entry.updated_at or 0) or int(time.time() * 1000)
    operator_name = (entry.operator_name or "").strip() or None

    cur = conn.execute(
        """
        SELECT operator_name, record_balance, updated_at
        FROM operators_top_records
        WHERE operator_id = ?
        """,
        (operator_id,),
    )
    row = cur.fetchone()
    if not row:
        if shift_balance <= 0 and not operator_name:
            return False
        conn.execute(
            """
            INSERT INTO operators_top_records (
                operator_id, operator_name, record_balance, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (operator_id, operator_name, max(0.0, shift_balance), updated_at),
        )
        return True

    current_balance = float(row["record_balance"] or 0.0)
    current_name = (row["operator_name"] or "").strip() or None
    current_updated = int(row["updated_at"] or 0)

    changed = False
    next_balance = current_balance
    if shift_balance > current_balance:
        next_balance = shift_balance
        changed = True

    next_name = current_name
    if operator_name and operator_name != current_name:
        next_name = operator_name
        changed = True

    if not changed:
        return False

    next_updated = max(current_updated, updated_at or int(time.time() * 1000))
    conn.execute(
        """
        UPDATE operators_top_records
        SET operator_name = ?,
            record_balance = ?,
            updated_at = ?
        WHERE operator_id = ?
        """,
        (next_name, max(0.0, next_balance), next_updated, operator_id),
    )
    return True


def upsert_actions_entry(conn: sqlite3.Connection, entry: TopActionEntry) -> bool:
    operator_id = entry.operator_id.strip()
    if not operator_id:
        return False
    day_key = normalize_top_day_key(entry.day_key)
    shift_actions = int(entry.shift_actions or 0)
    hour_actions = int(entry.hour_actions or 0)
    updated_at = int(entry.updated_at or 0) or int(time.time() * 1000)
    operator_name = (entry.operator_name or "").strip() or None

    cur = conn.execute(
        """
        SELECT operator_name, shift_actions, hour_actions, updated_at
        FROM operators_actions_top
        WHERE operator_id = ? AND day_key = ?
        """,
        (operator_id, day_key),
    )
    row = cur.fetchone()
    if not row:
        conn.execute(
            """
            INSERT INTO operators_actions_top (
                operator_id, day_key, operator_name,
                shift_actions, hour_actions, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (operator_id, day_key, operator_name, shift_actions, hour_actions, updated_at),
        )
        return shift_actions > 0 or hour_actions > 0 or bool(operator_name)

    current_shift = int(row["shift_actions"] or 0)
    current_hour = int(row["hour_actions"] or 0)
    current_updated = int(row["updated_at"] or 0)
    current_name = (row["operator_name"] or "").strip() or None

    changed = False
    next_shift = current_shift
    next_hour = current_hour

    if shift_actions > current_shift:
        next_shift = shift_actions
        changed = True
    if hour_actions > current_hour:
        next_hour = hour_actions
        changed = True

    next_name = current_name
    if operator_name and operator_name != current_name:
        next_name = operator_name
        changed = True

    if not changed:
        return False

    next_updated = max(current_updated, updated_at or int(time.time() * 1000))
    conn.execute(
        """
        UPDATE operators_actions_top
        SET operator_name = ?,
            shift_actions = ?,
            hour_actions = ?,
            updated_at = ?
        WHERE operator_id = ? AND day_key = ?
        """,
        (next_name, next_shift, next_hour, next_updated, operator_id, day_key),
    )
    return True


def upsert_action_record_entry(conn: sqlite3.Connection, entry: TopActionEntry) -> bool:
    operator_id = entry.operator_id.strip()
    if not operator_id:
        return False
    shift_actions = int(entry.shift_actions or 0)
    updated_at = int(entry.updated_at or 0) or int(time.time() * 1000)
    operator_name = (entry.operator_name or "").strip() or None

    cur = conn.execute(
        """
        SELECT operator_name, record_actions, updated_at
        FROM operators_actions_records
        WHERE operator_id = ?
        """,
        (operator_id,),
    )
    row = cur.fetchone()
    if not row:
        if shift_actions <= 0 and not operator_name:
            return False
        conn.execute(
            """
            INSERT INTO operators_actions_records (
                operator_id, operator_name, record_actions, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (operator_id, operator_name, max(0, shift_actions), updated_at),
        )
        return True

    current_record = int(row["record_actions"] or 0)
    current_name = (row["operator_name"] or "").strip() or None
    current_updated = int(row["updated_at"] or 0)

    changed = False
    next_record = current_record
    if shift_actions > current_record:
        next_record = shift_actions
        changed = True

    next_name = current_name
    if operator_name and operator_name != current_name:
        next_name = operator_name
        changed = True

    if not changed:
        return False

    next_updated = max(current_updated, updated_at or int(time.time() * 1000))
    conn.execute(
        """
        UPDATE operators_actions_records
        SET operator_name = ?,
            record_actions = ?,
            updated_at = ?
        WHERE operator_id = ?
        """,
        (next_name, max(0, next_record), next_updated, operator_id),
    )
    return True


def apply_profile_shift_delta(
    conn: sqlite3.Connection,
    entry: ProfileShiftDeltaEntry,
    day_key: str,
) -> bool:
    profile_id = entry.profile_id.strip()
    operator_id = entry.operator_id.strip()
    if not profile_id or not operator_id:
        return False
    operator_name = (entry.operator_name or "").strip() or None
    actions_total = int(entry.actions_total or 0)
    chat_count = int(entry.chat_count or 0)
    mail_count = int(entry.mail_count or 0)
    balance_earned = float(entry.balance_earned or 0.0)
    updated_at = int(entry.updated_at or 0) or int(time.time() * 1000)
    cur = conn.execute(
        """
        SELECT actions_total, chat_count, mail_count, balance_earned, updated_at, operator_name
        FROM profile_shift_stats
        WHERE profile_id = ? AND day_key = ? AND operator_id = ?
        """,
        (profile_id, day_key, operator_id),
    )
    row = cur.fetchone()
    if not row:
        conn.execute(
            """
            INSERT INTO profile_shift_stats (
                profile_id, day_key, operator_id, operator_name,
                actions_total, chat_count, mail_count, balance_earned, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                day_key,
                operator_id,
                operator_name,
                max(0, actions_total),
                max(0, chat_count),
                max(0, mail_count),
                balance_earned,
                updated_at,
            ),
        )
        return True
    current_updated = int(row["updated_at"] or 0)
    if updated_at < current_updated:
        return False
    current_actions = int(row["actions_total"] or 0)
    current_chat = int(row["chat_count"] or 0)
    current_mail = int(row["mail_count"] or 0)
    current_balance = float(row["balance_earned"] or 0.0)
    current_name = (row["operator_name"] or "").strip() or None
    start_ms, _ = get_operator_shift_day_range(day_key)
    is_new_shift = current_updated < start_ms and updated_at >= start_ms
    if is_new_shift:
        next_actions = max(0, actions_total)
        next_chat = max(0, chat_count)
        next_mail = max(0, mail_count)
        next_balance = max(0.0, balance_earned)
    else:
        next_actions = max(current_actions, actions_total)
        next_chat = max(current_chat, chat_count)
        next_mail = max(current_mail, mail_count)
        next_balance = max(current_balance, balance_earned)
    next_name = current_name
    if operator_name and operator_name != current_name:
        next_name = operator_name
    if (
        next_actions == current_actions
        and next_chat == current_chat
        and next_mail == current_mail
        and next_balance == current_balance
        and next_name == current_name
    ):
        return False
    conn.execute(
        """
        UPDATE profile_shift_stats
        SET actions_total = ?,
            chat_count = ?,
            mail_count = ?,
            balance_earned = ?,
            operator_name = ?,
            updated_at = ?
        WHERE profile_id = ? AND day_key = ? AND operator_id = ?
        """,
        (
            max(0, next_actions),
            max(0, next_chat),
            max(0, next_mail),
            next_balance,
            next_name,
            max(current_updated, updated_at),
            profile_id,
            day_key,
            operator_id,
        ),
    )
    return True


def fetch_profile_shift_stats(
    conn: sqlite3.Connection,
    day_key: str,
    profile_ids: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    ids = [pid.strip() for pid in profile_ids if pid and pid.strip()]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    params: List[Any] = [day_key, *ids]
    cur = conn.execute(
        f"""
        SELECT profile_id, operator_id, operator_name, actions_total, chat_count, mail_count, balance_earned, updated_at
        FROM profile_shift_stats
        WHERE day_key = ? AND profile_id IN ({placeholders})
        ORDER BY profile_id ASC, operator_id ASC
        """,
        params,
    )
    result: Dict[str, List[Dict[str, Any]]] = {}
    for row in cur.fetchall():
        pid = row["profile_id"]
        result.setdefault(pid, []).append(
            {
                "operator_id": row["operator_id"],
                "operator_name": (row["operator_name"] or "").strip(),
                "actions_total": int(row["actions_total"] or 0),
                "chat_count": int(row["chat_count"] or 0),
                "mail_count": int(row["mail_count"] or 0),
                "balance_earned": float(row["balance_earned"] or 0.0),
                "updated_at": int(row["updated_at"] or 0),
            }
        )
    return result


def upsert_operator_shift_summary(
    conn: sqlite3.Connection,
    payload: OperatorShiftSnapshotPayload,
) -> bool:
    operator_id = payload.operator_id.strip()
    if not operator_id:
        return False
    day_key = normalize_operator_shift_day_key(payload.day_key)
    operator_name = (payload.operator_name or "").strip() or None
    balance_total = float(payload.balance_total or 0.0)
    hour_actions_total = int(payload.hour_actions_total or 0)
    hour_chat_count = int(payload.hour_chat_count or 0)
    hour_mail_count = int(payload.hour_mail_count or 0)
    hour_start = int(payload.hour_start) if payload.hour_start is not None else None
    updated_at = int(payload.updated_at or 0) or int(time.time() * 1000)
    cur = conn.execute(
        """
        SELECT operator_name, balance_total, last_balance_total, last_balance_updated,
               actions_total, chat_count, mail_count,
               hour_actions_total, hour_chat_count, hour_mail_count, hour_start, updated_at
        FROM operator_shift_summary
        WHERE day_key = ? AND operator_id = ?
        """,
        (day_key, operator_id),
    )
    row = cur.fetchone()
    if not row:
        incoming_hour_start = int(hour_start or 0) if hour_start is not None else 0
        hourly_start_value = incoming_hour_start or ensure_hour_start(updated_at)
        conn.execute(
            """
            INSERT INTO operator_shift_summary (
                day_key, operator_id, operator_name, balance_total,
                last_balance_total, last_balance_updated,
                actions_total, chat_count, mail_count,
                hour_actions_total, hour_chat_count, hour_mail_count, hour_start,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                day_key,
                operator_id,
                operator_name,
                balance_total,
                balance_total,
                updated_at,
                0,
                0,
                0,
                max(0, hour_actions_total),
                max(0, hour_chat_count),
                max(0, hour_mail_count),
                hourly_start_value,
                updated_at,
            ),
        )
        if hourly_start_value:
            conn.execute(
                """
                INSERT INTO operator_hourly_balance (
                    day_key, operator_id, hour_start, balance_amount, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(day_key, operator_id, hour_start)
                DO UPDATE SET
                    balance_amount = excluded.balance_amount,
                    updated_at = excluded.updated_at
                """,
                (
                    day_key,
                    operator_id,
                    hourly_start_value,
                    max(0.0, balance_total),
                    updated_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO operator_hourly_actions (
                    day_key, operator_id, hour_start,
                    chat_count, mail_count, actions_total, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(day_key, operator_id, hour_start)
                DO UPDATE SET
                    chat_count = excluded.chat_count,
                    mail_count = excluded.mail_count,
                    actions_total = excluded.actions_total,
                    updated_at = excluded.updated_at
                """,
                (
                    day_key,
                    operator_id,
                    hourly_start_value,
                    max(0, hour_chat_count),
                    max(0, hour_mail_count),
                    max(0, hour_actions_total),
                    updated_at,
                ),
            )
            recompute_operator_shift_action_totals(conn, day_key, operator_id)
        return True
    current_updated = int(row["updated_at"] or 0)
    if updated_at < current_updated:
        return False
    current_name = (row["operator_name"] or "").strip() or None
    current_balance = float(row["balance_total"] or 0.0)
    last_balance_total = float(row["last_balance_total"] or 0.0)
    last_balance_updated = int(row["last_balance_updated"] or 0)
    current_actions = int(row["actions_total"] or 0)
    current_chat = int(row["chat_count"] or 0)
    current_mail = int(row["mail_count"] or 0)
    current_hour_actions = int(row["hour_actions_total"] or 0)
    current_hour_chat = int(row["hour_chat_count"] or 0)
    current_hour_mail = int(row["hour_mail_count"] or 0)
    current_hour_start = int(row["hour_start"] or 0)

    next_name = current_name
    if operator_name and operator_name != current_name:
        next_name = operator_name
    start_ms, _ = get_operator_shift_day_range(day_key)
    incoming_hour_start = int(hour_start or 0) if hour_start is not None else 0
    hourly_start_value = incoming_hour_start or ensure_hour_start(updated_at)
    is_new_shift = current_updated < start_ms and updated_at >= start_ms

    if is_new_shift:
        next_balance = max(0.0, balance_total)
        next_hour_actions = max(0, hour_actions_total)
        next_hour_chat = max(0, hour_chat_count)
        next_hour_mail = max(0, hour_mail_count)
        next_hour_start = hourly_start_value or current_hour_start
    else:
        next_balance = max(current_balance, balance_total)
        next_hour_actions = current_hour_actions
        next_hour_chat = current_hour_chat
        next_hour_mail = current_hour_mail
        next_hour_start = current_hour_start
    if not is_new_shift and hour_start is not None:
        if incoming_hour_start > current_hour_start:
            # New hour: reset hourly counters to incoming values (even if lower)
            next_hour_start = incoming_hour_start
            next_hour_actions = max(0, hour_actions_total)
            next_hour_chat = max(0, hour_chat_count)
            next_hour_mail = max(0, hour_mail_count)
        elif incoming_hour_start == current_hour_start:
            # Same hour: keep max logic
            next_hour_start = incoming_hour_start
            next_hour_actions = max(current_hour_actions, hour_actions_total)
            next_hour_chat = max(current_hour_chat, hour_chat_count)
            next_hour_mail = max(current_hour_mail, hour_mail_count)
    next_last_balance_total = last_balance_total
    next_last_balance_updated = last_balance_updated
    if is_new_shift or updated_at >= last_balance_updated:
        next_last_balance_total = max(0.0, balance_total)
        next_last_balance_updated = updated_at
    delta_balance = max(0.0, balance_total - last_balance_total)
    if is_new_shift:
        delta_balance = max(0.0, balance_total)
    changed = (
        next_balance != current_balance
        or next_hour_actions != current_hour_actions
        or next_hour_chat != current_hour_chat
        or next_hour_mail != current_hour_mail
        or next_hour_start != current_hour_start
        or next_name != current_name
        or next_last_balance_total != last_balance_total
        or next_last_balance_updated != last_balance_updated
    )
    conn.execute(
        """
        UPDATE operator_shift_summary
        SET operator_name = ?,
            balance_total = ?,
            last_balance_total = ?,
            last_balance_updated = ?,
            actions_total = ?,
            chat_count = ?,
            mail_count = ?,
            hour_actions_total = ?,
            hour_chat_count = ?,
            hour_mail_count = ?,
            hour_start = ?,
            updated_at = ?
        WHERE day_key = ? AND operator_id = ?
        """,
        (
            next_name,
            next_balance,
            next_last_balance_total,
            next_last_balance_updated,
            max(0, current_actions),
            max(0, current_chat),
            max(0, current_mail),
            max(0, next_hour_actions),
            max(0, next_hour_chat),
            max(0, next_hour_mail),
            next_hour_start,
            max(current_updated, updated_at),
            day_key,
            operator_id,
        ),
    )
    if hourly_start_value and delta_balance >= 0:
        cur = conn.execute(
            """
            SELECT balance_amount
            FROM operator_hourly_balance
            WHERE day_key = ? AND operator_id = ? AND hour_start = ?
            """,
            (day_key, operator_id, hourly_start_value),
        )
        row_balance = cur.fetchone()
        current_hour_balance = (
            float(row_balance["balance_amount"] or 0.0) if row_balance else 0.0
        )
        next_hour_balance = current_hour_balance + delta_balance
        conn.execute(
            """
            INSERT INTO operator_hourly_balance (
                day_key, operator_id, hour_start, balance_amount, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(day_key, operator_id, hour_start)
            DO UPDATE SET
                balance_amount = excluded.balance_amount,
                updated_at = excluded.updated_at
            """,
            (
                day_key,
                operator_id,
                hourly_start_value,
                max(0.0, next_hour_balance),
                updated_at,
            ),
        )
    if hourly_start_value:
        conn.execute(
            """
            INSERT INTO operator_hourly_actions (
                day_key, operator_id, hour_start,
                chat_count, mail_count, actions_total, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(day_key, operator_id, hour_start)
            DO UPDATE SET
                chat_count = excluded.chat_count,
                mail_count = excluded.mail_count,
                actions_total = excluded.actions_total,
                updated_at = excluded.updated_at
            """,
            (
                day_key,
                operator_id,
                hourly_start_value,
                max(0, next_hour_chat),
                max(0, next_hour_mail),
                max(0, next_hour_actions),
                updated_at,
            ),
        )
    recompute_changed = recompute_operator_shift_action_totals(conn, day_key, operator_id)
    return changed or recompute_changed


def get_operator_shift_summary(
    conn: sqlite3.Connection,
    day_key: str,
    operator_id: str,
) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT day_key, operator_id, operator_name, balance_total,
               actions_total, chat_count, mail_count,
               hour_actions_total, hour_chat_count, hour_mail_count, hour_start,
               updated_at
        FROM operator_shift_summary
        WHERE day_key = ? AND operator_id = ?
        """,
        (day_key, operator_id),
    )
    row = cur.fetchone()
    if not row:
        return None
    fallback_name = ""
    try:
        cur_name = conn.execute(
            """
            SELECT payload
            FROM operator_state
            WHERE operator_id = ? AND day_key = ? AND section = ?
            """,
            ("__GLOBAL__", GLOBAL_OPERATOR_NAMES_DAY_KEY, "operator_names"),
        ).fetchone()
        if cur_name and cur_name["payload"]:
            raw_payload = cur_name["payload"]
            if isinstance(raw_payload, str) and raw_payload.strip():
                payload = json.loads(raw_payload)
                if isinstance(payload, list):
                    for entry in payload:
                        if str(entry.get("operator_id") or "").strip() == operator_id:
                            fallback_name = (entry.get("operator_name") or "").strip()
                            if fallback_name:
                                break
    except Exception:
        fallback_name = ""
    hourly_cur = conn.execute(
        """
        SELECT hour_start, balance_amount, updated_at
        FROM operator_hourly_balance
        WHERE day_key = ? AND operator_id = ?
        ORDER BY hour_start ASC
        """,
        (day_key, operator_id),
    )
    hourly_rows = [
        {
            "hour_start": int(r["hour_start"] or 0),
            "balance_amount": float(r["balance_amount"] or 0.0),
            "updated_at": int(r["updated_at"] or 0),
        }
        for r in hourly_cur.fetchall()
    ]
    hourly_actions_cur = conn.execute(
        """
        SELECT hour_start, chat_count, mail_count, actions_total, updated_at
        FROM operator_hourly_actions
        WHERE day_key = ? AND operator_id = ?
        ORDER BY hour_start ASC
        """,
        (day_key, operator_id),
    )
    hourly_actions_rows = [
        {
            "hour_start": int(r["hour_start"] or 0),
            "chat_count": int(r["chat_count"] or 0),
            "mail_count": int(r["mail_count"] or 0),
            "actions_total": int(r["actions_total"] or 0),
            "updated_at": int(r["updated_at"] or 0),
        }
        for r in hourly_actions_cur.fetchall()
    ]
    return {
        "day_key": row["day_key"],
        "operator_id": row["operator_id"],
        "operator_name": (row["operator_name"] or "").strip() or fallback_name,
        "balance_total": float(row["balance_total"] or 0.0),
        "actions_total": int(row["actions_total"] or 0),
        "chat_count": int(row["chat_count"] or 0),
        "mail_count": int(row["mail_count"] or 0),
        "hour_actions_total": int(row["hour_actions_total"] or 0),
        "hour_chat_count": int(row["hour_chat_count"] or 0),
        "hour_mail_count": int(row["hour_mail_count"] or 0),
        "hour_start": int(row["hour_start"] or 0),
        "updated_at": int(row["updated_at"] or 0),
        "hourly_balance": hourly_rows,
        "hourly_actions": hourly_actions_rows,
    }


def fetch_state_sections(
    conn: sqlite3.Connection,
    operator_id: str,
    day_key: str,
    section_filter: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    params: List[Any] = [operator_id, day_key]
    query = """
        SELECT section, updated_at, payload
        FROM operator_state
        WHERE operator_id = ? AND day_key = ?
    """
    if section_filter:
        placeholders = ",".join("?" for _ in section_filter)
        query += f" AND section IN ({placeholders})"
        params.extend(section_filter)
    cur = conn.execute(query, params)
    result: Dict[str, Dict[str, Any]] = {}
    for row in cur.fetchall():
        payload = {}
        raw_payload = row["payload"]
        if isinstance(raw_payload, str) and raw_payload.strip():
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                payload = {}
        result[row["section"]] = {
            "updated_at": int(row["updated_at"] or 0),
            "data": payload,
        }
    return result


def apply_section_side_effects(
    conn: sqlite3.Connection,
    section: str,
    data: Any,
    default_shift: Optional[str],
) -> None:
    if section == "reports" and isinstance(data, list):
        for entry in data:
            try:
                report = ReportPayload.parse_obj(entry)
            except Exception:
                continue
            try:
                upsert_report(conn, report)
            except Exception:
                continue
    elif section == "hourly_stats" and isinstance(data, list):
        for entry in data:
            try:
                stat = HourlyStatPayload.parse_obj(entry)
            except Exception:
                continue
            try:
                upsert_hourly_stat(conn, stat, default_shift)
            except Exception:
                continue
    elif section == "history" and isinstance(data, dict):
        upsert_history_hourly_profiles(conn, data, default_shift)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if math.isfinite(num):
        return num
    return default


def upsert_history_hourly_profiles(
    conn: sqlite3.Connection,
    data: Dict[str, Any],
    default_shift: Optional[str],
) -> None:
    entries = data.get("hourlyProfiles")
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        female_id = str(
            entry.get("female_id")
            or entry.get("profile_id")
            or entry.get("profile_uid")
            or ""
        ).strip()
        operator_id = str(
            entry.get("operator_id") or entry.get("operatorId") or ""
        ).strip()
        hour_start = safe_int(entry.get("hour_start") or entry.get("start"))
        if not female_id or not operator_id or hour_start <= 0:
            continue
        paid_chat = safe_int(entry.get("paid_chat"))
        paid_mail = safe_int(entry.get("paid_mail"))
        paid_actions_raw = entry.get("paid_actions")
        if paid_actions_raw is None:
            paid_actions = paid_chat + paid_mail
        else:
            paid_actions = safe_int(paid_actions_raw)
        try:
            stat = HourlyStatPayload(
                female_id=female_id,
                operator_id=operator_id,
                operator_name=entry.get("operator_name")
                or entry.get("operatorName")
                or None,
                hour_start=hour_start,
                actions_total=safe_int(entry.get("actions_total") or entry.get("total")),
                actions_paid=paid_actions,
                balance_earned=safe_float(
                    entry.get("balance_earned") or entry.get("balance")
                ),
                chat_count=safe_int(entry.get("chat_count") or entry.get("chat")),
                mail_count=safe_int(entry.get("mail_count") or entry.get("mail")),
                paid_chat=paid_chat,
                paid_mail=paid_mail,
            )
        except Exception:
            continue
        try:
            upsert_hourly_stat(conn, stat, default_shift)
        except Exception:
            continue


def build_history_hourly_global(
    conn: sqlite3.Connection,
    day_key: str,
    operator_id: str,
) -> List[Dict[str, Any]]:
    if not operator_id:
        return []
    start_ms, end_ms = get_kyiv_day_range(day_key)
    cur = conn.execute(
        """
        SELECT
            hour_start,
            SUM(actions_total) AS actions_total,
            SUM(chat_count) AS chat_count,
            SUM(mail_count) AS mail_count,
            SUM(balance_earned) AS balance_earned,
            SUM(paid_chat) AS paid_chat,
            SUM(paid_mail) AS paid_mail
        FROM hourly_stats
        WHERE hour_start >= ? AND hour_start < ? AND operator_id = ?
        GROUP BY hour_start
        ORDER BY hour_start ASC
        """,
        (start_ms, end_ms, operator_id),
    )
    rows: List[Dict[str, Any]] = []
    for row in cur.fetchall():
        hour_start = safe_int(row["hour_start"])
        if hour_start <= 0:
            continue
        paid_chat = safe_int(row["paid_chat"])
        paid_mail = safe_int(row["paid_mail"])
        rows.append(
            {
                "hour_start": hour_start,
                "actions_total": safe_int(row["actions_total"]),
                "chat_count": safe_int(row["chat_count"]),
                "mail_count": safe_int(row["mail_count"]),
                "balance_earned": safe_float(row["balance_earned"]),
                "paid_chat": paid_chat,
                "paid_mail": paid_mail,
                "paid_actions": paid_chat + paid_mail,
                "day_key": day_key,
                "operator_id": operator_id,
            }
        )
    return rows


def build_history_hourly_profiles(
    conn: sqlite3.Connection,
    day_key: str,
) -> List[Dict[str, Any]]:
    start_ms, end_ms = get_kyiv_day_range(day_key)
    cur = conn.execute(
        """
        SELECT
            female_id,
            operator_id,
            operator_name,
            hour_start,
            actions_total,
            actions_paid,
            balance_earned,
            chat_count,
            mail_count,
            paid_chat,
            paid_mail
        FROM hourly_stats
        WHERE hour_start >= ? AND hour_start < ?
        ORDER BY female_id ASC, hour_start DESC, operator_id ASC
        """,
        (start_ms, end_ms),
    )
    rows = cur.fetchall()
    summaries: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        female_id = (row["female_id"] or "").strip()
        if not female_id:
            continue
        summary = summaries.setdefault(
            female_id,
            {
                "female_id": female_id,
                "shift_key": day_key,
                "balance_earned": 0.0,
                "actions_total": 0,
                "actions_paid": 0,
                "chat_count": 0,
                "mail_count": 0,
                "operator_summary": {},
            },
        )
        balance_value = safe_float(row["balance_earned"])
        summary["balance_earned"] += balance_value
        actions_total_value = safe_int(row["actions_total"])
        summary["actions_total"] += actions_total_value
        paid_actions_value = safe_int(row["actions_paid"])
        if paid_actions_value <= 0:
            paid_actions_value = safe_int(row["paid_chat"]) + safe_int(row["paid_mail"])
        summary["actions_paid"] += paid_actions_value
        summary["chat_count"] += safe_int(row["chat_count"])
        summary["mail_count"] += safe_int(row["mail_count"])
        op_id = (row["operator_id"] or "").strip()
        op_name = (row["operator_name"] or "").strip()
        operator_summary = summary["operator_summary"].setdefault(
            op_id,
            {
                "operator_id": op_id,
                "operator_name": op_name,
                "actions_total": 0,
                "balance_earned": 0.0,
            },
        )
        operator_summary["operator_name"] = op_name
        operator_summary["actions_total"] += actions_total_value
        operator_summary["balance_earned"] += balance_value
    for summary in summaries.values():
        operator_entries = list(summary["operator_summary"].values())
        summary["operator_summary"] = operator_entries
    payload: List[Dict[str, Any]] = []
    attached_summary: Dict[str, bool] = defaultdict(bool)
    for row in rows:
        female_id = (row["female_id"] or "").strip()
        if not female_id:
            continue
        hour_start = safe_int(row["hour_start"])
        if hour_start <= 0:
            continue
        paid_chat = safe_int(row["paid_chat"])
        paid_mail = safe_int(row["paid_mail"])
        entry: Dict[str, Any] = {
            "female_id": female_id,
            "profile_uid": female_id,
            "hour_start": hour_start,
            "actions_total": safe_int(row["actions_total"]),
            "chat_count": safe_int(row["chat_count"]),
            "mail_count": safe_int(row["mail_count"]),
            "balance_earned": safe_float(row["balance_earned"]),
            "paid_chat": paid_chat,
            "paid_mail": paid_mail,
            "paid_actions": paid_chat + paid_mail,
            "operator_id": (row["operator_id"] or "").strip(),
            "operator_name": (row["operator_name"] or "").strip(),
            "day_key": day_key,
        }
        summary = summaries.get(female_id)
        if summary and not attached_summary[female_id]:
            entry["shift_summary"] = summary
            attached_summary[female_id] = True
        payload.append(entry)
    return payload


def build_monitor_snapshot(
    conn: sqlite3.Connection,
    operator_id: str,
    day_key: str,
) -> Dict[str, Any]:
    if not operator_id:
        return {"counts": {"chat": 0, "mail": 0, "hourHistory": []}}
    start_ms, end_ms = get_kyiv_day_range(day_key)
    cur = conn.execute(
        """
        SELECT
            hour_start,
            SUM(chat_count) AS chat_count,
            SUM(mail_count) AS mail_count,
            SUM(actions_total) AS actions_total
        FROM hourly_stats
        WHERE operator_id = ?
          AND hour_start >= ?
          AND hour_start < ?
        GROUP BY hour_start
        ORDER BY hour_start DESC
        """,
        (operator_id, start_ms, end_ms),
    )
    rows = cur.fetchall()
    total_chat = 0
    total_mail = 0
    hour_history: List[Dict[str, Any]] = []
    hour_record = 0
    current_hour_start = None
    current_hour_total = 0
    current_hour_chat = 0
    current_hour_mail = 0
    for idx, row in enumerate(rows):
        hour_start = safe_int(row["hour_start"])
        if hour_start <= 0:
            continue
        chat_value = safe_int(row["chat_count"])
        mail_value = safe_int(row["mail_count"])
        total_value = safe_int(row["actions_total"])
        total_chat += chat_value
        total_mail += mail_value
        hour_record = max(hour_record, total_value)
        if idx == 0:
            current_hour_start = hour_start
            current_hour_total = total_value
            current_hour_chat = chat_value
            current_hour_mail = mail_value
        hour_history.append(
            {
                "start": hour_start,
                "total": total_value,
                "chat": chat_value,
                "mail": mail_value,
            }
        )
    counts = {
        "chat": total_chat,
        "mail": total_mail,
        "hourStart": current_hour_start,
        "hourTotal": current_hour_total,
        "hourChat": current_hour_chat,
        "hourMail": current_hour_mail,
        "hourHistory": hour_history,
        "hourRecord": hour_record,
    }
    return {
        "counts": counts,
        "goal": 0,
        "hourGoal": hour_record,
    }


def enrich_history_section_with_hourly(
    conn: sqlite3.Connection,
    payload: Any,
    day_key: str,
    operator_id: str,
) -> Dict[str, Any]:
    base = payload if isinstance(payload, dict) else {}
    enriched = dict(base)
    enriched["hourlyGlobal"] = build_history_hourly_global(conn, day_key, operator_id)
    enriched["hourlyProfiles"] = build_history_hourly_profiles(conn, day_key)
    current_day_key = normalize_state_day_key(None)
    if day_key == current_day_key and not enriched.get("monitor"):
        enriched["monitor"] = build_monitor_snapshot(conn, operator_id, day_key)
    return enriched


def get_count_from_db(male_id: str) -> int:
  """
  Возвращает количество сохранённых отчётов по male_id
  из таблицы reports в основной БД.
  """
  if not os.path.exists(DB_PATH):
      raise HTTPException(status_code=500, detail="DB file not found")
  try:
      conn = get_conn()
      cur = conn.cursor()
      cur.execute(
          "SELECT COUNT(*) AS c FROM reports WHERE male_id = ?",
          (male_id,),
      )
      row = cur.fetchone()
      conn.close()
      return int(row["c"] if row and row["c"] is not None else 0)
  except Exception as e:  # pragma: no cover - surface error for API response
      raise HTTPException(status_code=500, detail=f"DB error: {e}")


def active_extension_passwords_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM extension_passwords
        WHERE deleted_at IS NULL AND is_active = 1
        """
    ).fetchone()
    if not row:
        return 0
    return int(row["c"] or 0)


def fetch_active_extension_password(
    conn: sqlite3.Connection, password: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, name, password, team_name
        FROM extension_passwords
        WHERE deleted_at IS NULL
          AND is_active = 1
          AND password = ?
        LIMIT 1
        """,
        (password,),
    ).fetchone()


def upsert_operator_team_state(
    conn: sqlite3.Connection,
    operator_id: str,
    team_name: str,
    password_id: int,
    now_ms: int,
) -> None:
    operator_key = operator_id.strip()
    team_key = team_name.strip()
    if not operator_key or not team_key:
        return
    conn.execute(
        """
        INSERT INTO operator_team_state (
            operator_id,
            team_name,
            password_id,
            updated_at
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(operator_id) DO UPDATE SET
            team_name = excluded.team_name,
            password_id = excluded.password_id,
            updated_at = excluded.updated_at
        """,
        (operator_key, team_key, int(password_id), now_ms),
    )


def fetch_operator_team_names(
    conn: sqlite3.Connection, operator_ids: List[str]
) -> Dict[str, str]:
    keys = [str(item or "").strip() for item in operator_ids if str(item or "").strip()]
    if not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"""
        SELECT operator_id, team_name
        FROM operator_team_state
        WHERE operator_id IN ({placeholders})
        """,
        tuple(keys),
    ).fetchall()
    out: Dict[str, str] = {}
    for row in rows:
        operator_id = str(row["operator_id"] or "").strip()
        if not operator_id:
            continue
        out[operator_id] = str(row["team_name"] or "").strip()
    return out


def fetch_install_admin_flags(
    conn: sqlite3.Connection, install_ids: List[str]
) -> Dict[str, bool]:
    keys = list(
        {
            str(item or "").strip()
            for item in install_ids
            if str(item or "").strip()
        }
    )
    if not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"""
        SELECT install_id, is_admin_install
        FROM extension_install_registry
        WHERE install_id IN ({placeholders})
        """,
        tuple(keys),
    ).fetchall()
    out: Dict[str, bool] = {}
    for row in rows:
        install_id = str(row["install_id"] or "").strip()
        if not install_id:
            continue
        out[install_id] = bool(int(row["is_admin_install"] or 0))
    return out


def fetch_admin_install_name_meta(
    conn: sqlite3.Connection,
) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            eir.install_id,
            COALESCE(
                NULLIF(TRIM(eir.created_with_version), ''),
                NULLIF(TRIM(eir.last_seen_version), ''),
                ''
            ) AS install_version,
            eir.first_seen_at,
            eir.last_seen_at,
            COALESCE((
                SELECT TRIM(p.name)
                FROM extension_password_usages u
                INNER JOIN extension_passwords p
                    ON p.id = u.password_id
                WHERE u.install_id = eir.install_id
                  AND p.name IS NOT NULL
                  AND TRIM(p.name) <> ''
                ORDER BY u.first_used_at ASC, u.id ASC
                LIMIT 1
            ), '') AS password_name
        FROM extension_install_registry eir
        WHERE eir.is_admin_install = 1
        """
    ).fetchall()
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        install_id = str(row["install_id"] or "").strip()
        if not install_id:
            continue
        version_key = str(row["install_version"] or "").strip()
        grouped.setdefault(version_key, []).append(
            {
                "install_id": install_id,
                "install_version": version_key,
                "first_seen_at": int(row["first_seen_at"] or 0),
                "last_seen_at": int(row["last_seen_at"] or 0),
                "password_name": str(row["password_name"] or "").strip(),
            }
        )
    meta: Dict[str, Dict[str, Any]] = {}
    for version_key, items in grouped.items():
        ordered_items = sorted(
            items,
            key=lambda entry: (
                int(entry["first_seen_at"] or 0),
                int(entry["last_seen_at"] or 0),
                str(entry["install_id"] or ""),
            ),
        )
        for index, entry in enumerate(ordered_items, start=1):
            password_name = str(entry.get("password_name") or "").strip()
            meta[str(entry["install_id"])] = {
                "admin_name": password_name or f"Админ {index}",
                "install_version": version_key,
                "first_seen_at": int(entry["first_seen_at"] or 0),
                "last_seen_at": int(entry["last_seen_at"] or 0),
                "admin_index": index,
                "password_name": password_name,
            }
    return meta


def build_admin_name_list(
    install_ids: List[str], admin_install_meta: Dict[str, Dict[str, Any]]
) -> List[str]:
    ordered_meta: List[Dict[str, Any]] = []
    seen_install_ids: set[str] = set()
    for raw_install_id in install_ids:
        install_id = str(raw_install_id or "").strip()
        if not install_id or install_id in seen_install_ids:
            continue
        meta = admin_install_meta.get(install_id)
        if not meta:
            continue
        seen_install_ids.add(install_id)
        ordered_meta.append(meta)
    if not ordered_meta:
        return []
    name_counts: Dict[str, int] = {}
    for meta in ordered_meta:
        admin_name = str(meta.get("admin_name") or "").strip()
        if not admin_name:
            continue
        name_counts[admin_name] = name_counts.get(admin_name, 0) + 1
    labels: List[str] = []
    for meta in ordered_meta:
        admin_name = str(meta.get("admin_name") or "").strip()
        if not admin_name:
            continue
        version = str(meta.get("install_version") or "").strip()
        if name_counts.get(admin_name, 0) > 1 and version:
            labels.append(f"{admin_name} ({version})")
        else:
            labels.append(admin_name)
    return labels


def fetch_related_admin_names_by_operator(
    conn: sqlite3.Connection,
    operator_ids: List[str],
    admin_install_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    keys = [str(item or "").strip() for item in operator_ids if str(item or "").strip()]
    if not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"""
        SELECT ioh.operator_id, ioh.install_id
        FROM install_operator_history ioh
        INNER JOIN extension_install_registry eir
            ON eir.install_id = ioh.install_id
        WHERE ioh.operator_id IN ({placeholders})
          AND eir.is_admin_install = 1
        GROUP BY ioh.operator_id, ioh.install_id
        ORDER BY eir.first_seen_at DESC, eir.last_seen_at DESC, ioh.install_id ASC
        """,
        tuple(keys),
    ).fetchall()
    install_ids_by_operator: Dict[str, List[str]] = {}
    for row in rows:
        operator_id = str(row["operator_id"] or "").strip()
        install_id = str(row["install_id"] or "").strip()
        if not operator_id or not install_id:
            continue
        install_ids_by_operator.setdefault(operator_id, []).append(install_id)
    return {
        operator_id: build_admin_name_list(install_ids, admin_install_meta)
        for operator_id, install_ids in install_ids_by_operator.items()
    }


def fetch_related_admin_names_by_agency(
    conn: sqlite3.Connection,
    agency_ids: List[str],
    admin_install_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    keys = [str(item or "").strip() for item in agency_ids if str(item or "").strip()]
    if not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"""
        SELECT ioh.agency_id, ioh.install_id
        FROM install_operator_history ioh
        INNER JOIN extension_install_registry eir
            ON eir.install_id = ioh.install_id
        WHERE ioh.agency_id IN ({placeholders})
          AND eir.is_admin_install = 1
        GROUP BY ioh.agency_id, ioh.install_id
        ORDER BY eir.first_seen_at DESC, eir.last_seen_at DESC, ioh.install_id ASC
        """,
        tuple(keys),
    ).fetchall()
    install_ids_by_agency: Dict[str, List[str]] = {}
    for row in rows:
        agency_id = str(row["agency_id"] or "").strip()
        install_id = str(row["install_id"] or "").strip()
        if not agency_id or not install_id:
            continue
        install_ids_by_agency.setdefault(agency_id, []).append(install_id)
    return {
        agency_id: build_admin_name_list(install_ids, admin_install_meta)
        for agency_id, install_ids in install_ids_by_agency.items()
    }


def fetch_multi_install_current_version_meta(
    conn: sqlite3.Connection, operator_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    version_key = str(LATEST_EXTENSION_VERSION or "").strip()
    keys = [str(item or "").strip() for item in operator_ids if str(item or "").strip()]
    if not version_key or not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"""
        SELECT
            history.operator_id,
            COUNT(*) AS install_count,
            GROUP_CONCAT(history.install_id) AS install_ids
        FROM (
            SELECT DISTINCT operator_id, install_id
            FROM operator_install_binding_history
            WHERE extension_version = ?
              AND operator_id IN ({placeholders})
        ) AS history
        GROUP BY history.operator_id
        HAVING COUNT(*) >= 2
        """,
        (version_key, *keys),
    ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        operator_id = str(row["operator_id"] or "").strip()
        if not operator_id:
            continue
        install_ids = [
            install_id.strip()
            for install_id in str(row["install_ids"] or "").split(",")
            if install_id and install_id.strip()
        ]
        out[operator_id] = {
            "multi_install_current_version": True,
            "multi_install_current_version_count": int(row["install_count"] or 0),
            "multi_install_current_version_install_ids": install_ids,
            "multi_install_current_version_version": version_key,
        }
    return out


def fetch_admin_install_summary_items(
    conn: sqlite3.Connection, agency_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    admin_install_meta = fetch_admin_install_name_meta(conn)
    params: List[Any] = []
    agency_filter_sql = ""
    if agency_id is not None:
        agency_key = str(agency_id or "").strip()
        agency_filter_sql = """
          AND EXISTS (
            SELECT 1
            FROM install_operator_history ioh
            WHERE ioh.install_id = eir.install_id
              AND ioh.agency_id = ?
          )
        """
        params.append(agency_key)
    rows = conn.execute(
        f"""
        SELECT
            eir.install_id,
            eir.created_with_version,
            eir.last_seen_version,
            eir.first_operator_id,
            eir.current_operator_id,
            eir.admin_reason,
            eir.first_seen_at,
            eir.last_seen_at,
            (
                SELECT GROUP_CONCAT(sorted.operator_id)
                FROM (
                    SELECT DISTINCT ioh2.operator_id AS operator_id
                    FROM install_operator_history ioh2
                    WHERE ioh2.install_id = eir.install_id
                    ORDER BY ioh2.operator_id ASC
                ) AS sorted
            ) AS operators,
            (
                SELECT GROUP_CONCAT(sorted_agencies.agency_id)
                FROM (
                    SELECT DISTINCT TRIM(ioh3.agency_id) AS agency_id
                    FROM install_operator_history ioh3
                    WHERE ioh3.install_id = eir.install_id
                      AND ioh3.agency_id IS NOT NULL
                      AND TRIM(ioh3.agency_id) <> ''
                    ORDER BY agency_id ASC
                ) AS sorted_agencies
            ) AS agency_ids
        FROM extension_install_registry eir
        WHERE eir.is_admin_install = 1
        {agency_filter_sql}
        ORDER BY eir.first_seen_at DESC, eir.last_seen_at DESC, eir.install_id ASC
        """,
        tuple(params),
    ).fetchall()
    operator_ids: List[str] = []
    for row in rows:
        operator_ids.extend(
            [
                operator_id.strip()
                for operator_id in str(row["operators"] or "").split(",")
                if operator_id and operator_id.strip()
            ]
        )
    team_names_by_operator = fetch_operator_team_names(conn, operator_ids)
    items: List[Dict[str, Any]] = []
    for row in rows:
        install_id = str(row["install_id"] or "").strip()
        if not install_id:
            continue
        operators = [
            operator_id.strip()
            for operator_id in str(row["operators"] or "").split(",")
            if operator_id and operator_id.strip()
        ]
        meta = admin_install_meta.get(install_id, {})
        items.append(
            {
                "install_id": install_id,
                "install_short": install_id[:8],
                "install_version": str(
                    row["last_seen_version"] or row["created_with_version"] or ""
                ).strip(),
                "admin_name": str(meta.get("admin_name") or "").strip(),
                "agency_ids": [
                    agency_key.strip()
                    for agency_key in str(row["agency_ids"] or "").split(",")
                    if agency_key and agency_key.strip()
                ],
                "team_names": ", ".join(
                    sorted(
                        {
                            team_name
                            for team_name in (
                                team_names_by_operator.get(operator_id, "")
                                for operator_id in operators
                            )
                            if team_name
                        }
                    )
                ),
                "first_operator_id": str(row["first_operator_id"] or "").strip(),
                "current_operator_id": str(row["current_operator_id"] or "").strip(),
                "admin_reason": str(row["admin_reason"] or "").strip(),
                "first_seen_at": int(row["first_seen_at"] or 0),
                "last_seen_at": int(row["last_seen_at"] or 0),
                "operators_history": operators,
                "is_admin_install": True,
            }
        )
    return items


def upsert_extension_password_usage(
    conn: sqlite3.Connection,
    password_id: int,
    install_id: str,
    now_ms: int,
    extension_version: Optional[str] = None,
    count_success: bool = True,
) -> None:
    if not install_id:
        return
    version_value = (extension_version or "").strip() or None
    success_delta = 0
    conn.execute(
        """
        INSERT INTO extension_password_usages (
            password_id,
            install_id,
            extension_version,
            first_used_at,
            last_used_at,
            success_count
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(password_id, install_id) DO UPDATE SET
            extension_version = CASE
                WHEN excluded.extension_version IS NOT NULL
                     AND TRIM(excluded.extension_version) <> ''
                THEN excluded.extension_version
                ELSE extension_password_usages.extension_version
            END,
            last_used_at = excluded.last_used_at,
            success_count = extension_password_usages.success_count + ?
        """,
        (
            password_id,
            install_id,
            version_value,
            now_ms,
            now_ms,
            success_delta,
            success_delta,
        ),
    )


def upsert_extension_password_operator_usage(
    conn: sqlite3.Connection,
    password_id: int,
    install_id: str,
    operator_id: str,
    agency_id: Optional[str],
    now_ms: int,
    count_success: bool = True,
) -> None:
    install_key = install_id.strip()
    operator_key = operator_id.strip()
    agency_key = (agency_id or "").strip() or None
    if not install_key or not operator_key:
        return
    success_delta = 0
    conn.execute(
        """
        INSERT INTO extension_password_usage_operators (
            password_id,
            install_id,
            operator_id,
            agency_id,
            first_used_at,
            last_used_at,
            success_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(password_id, install_id, operator_id) DO UPDATE SET
            agency_id = CASE
                WHEN excluded.agency_id IS NOT NULL AND TRIM(excluded.agency_id) <> ''
                THEN excluded.agency_id
                ELSE extension_password_usage_operators.agency_id
            END,
            last_used_at = excluded.last_used_at,
            success_count = extension_password_usage_operators.success_count + ?
        """,
        (
            password_id,
            install_key,
            operator_key,
            agency_key,
            now_ms,
            now_ms,
            success_delta,
            success_delta,
        ),
    )


def fetch_extension_install_registry(
    conn: sqlite3.Connection, install_id: str
) -> Optional[sqlite3.Row]:
    install_key = str(install_id or "").strip()
    if not install_key:
        return None
    return conn.execute(
        """
        SELECT *
        FROM extension_install_registry
        WHERE install_id = ?
        LIMIT 1
        """,
        (install_key,),
    ).fetchone()


def upsert_extension_install_registry(
    conn: sqlite3.Connection,
    install_id: str,
    now_ms: int,
    extension_version: Optional[str] = None,
    operator_id: Optional[str] = None,
    agency_id: Optional[str] = None,
) -> sqlite3.Row:
    install_key = str(install_id or "").strip()
    operator_key = str(operator_id or "").strip() or None
    agency_key = str(agency_id or "").strip() or None
    version_key = str(extension_version or "").strip() or None
    if not install_key:
        raise ValueError("install_id is required")
    existing = fetch_extension_install_registry(conn, install_key)
    if not existing:
        conn.execute(
            """
            INSERT INTO extension_install_registry (
                install_id,
                first_seen_at,
                last_seen_at,
                created_with_version,
                last_seen_version,
                is_admin_install,
                admin_reason,
                first_operator_id,
                current_operator_id,
                agency_id,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?)
            """,
            (
                install_key,
                now_ms,
                now_ms,
                version_key,
                version_key,
                operator_key,
                operator_key,
                agency_key,
                now_ms,
            ),
        )
    else:
        conn.execute(
            """
            UPDATE extension_install_registry
            SET
                last_seen_at = ?,
                last_seen_version = CASE
                    WHEN ? IS NOT NULL AND TRIM(?) <> '' THEN ?
                    ELSE last_seen_version
                END,
                current_operator_id = CASE
                    WHEN ? IS NOT NULL AND TRIM(?) <> '' THEN ?
                    ELSE current_operator_id
                END,
                agency_id = CASE
                    WHEN ? IS NOT NULL AND TRIM(?) <> '' THEN ?
                    ELSE agency_id
                END,
                updated_at = ?
            WHERE install_id = ?
            """,
            (
                now_ms,
                version_key,
                version_key,
                version_key,
                operator_key,
                operator_key,
                operator_key,
                agency_key,
                agency_key,
                agency_key,
                now_ms,
                install_key,
            ),
        )
    row = fetch_extension_install_registry(conn, install_key)
    if not row:
        raise ValueError("install registry row was not created")
    return row


def upsert_install_operator_history(
    conn: sqlite3.Connection,
    install_id: str,
    operator_id: str,
    agency_id: Optional[str],
    now_ms: int,
    is_admin_context: bool = False,
) -> None:
    install_key = str(install_id or "").strip()
    operator_key = str(operator_id or "").strip()
    agency_key = str(agency_id or "").strip() or None
    if not install_key or not operator_key:
        return
    admin_seen = 1 if is_admin_context else 0
    conn.execute(
        """
        INSERT INTO install_operator_history (
            install_id,
            operator_id,
            agency_id,
            first_seen_at,
            last_seen_at,
            seen_count,
            is_admin_context_seen
        )
        VALUES (?, ?, ?, ?, ?, 1, ?)
        ON CONFLICT(install_id, operator_id) DO UPDATE SET
            agency_id = CASE
                WHEN excluded.agency_id IS NOT NULL AND TRIM(excluded.agency_id) <> ''
                THEN excluded.agency_id
                ELSE install_operator_history.agency_id
            END,
            last_seen_at = excluded.last_seen_at,
            seen_count = install_operator_history.seen_count + 1,
            is_admin_context_seen = CASE
                WHEN excluded.is_admin_context_seen = 1 THEN 1
                ELSE install_operator_history.is_admin_context_seen
            END
        """,
        (
            install_key,
            operator_key,
            agency_key,
            now_ms,
            now_ms,
            admin_seen,
        ),
    )


def count_distinct_install_operator_ids(
    conn: sqlite3.Connection, install_id: str
) -> int:
    install_key = str(install_id or "").strip()
    if not install_key:
        return 0
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT operator_id) AS total
        FROM install_operator_history
        WHERE install_id = ?
        """,
        (install_key,),
    ).fetchone()
    return int(row["total"] or 0) if row else 0


def mark_install_as_admin(
    conn: sqlite3.Connection,
    install_id: str,
    now_ms: int,
    reason: str,
    operator_id: Optional[str] = None,
    agency_id: Optional[str] = None,
    extension_version: Optional[str] = None,
) -> sqlite3.Row:
    install_key = str(install_id or "").strip()
    operator_key = str(operator_id or "").strip() or None
    agency_key = str(agency_id or "").strip() or None
    version_key = str(extension_version or "").strip() or None
    if not install_key:
        raise ValueError("install_id is required")
    conn.execute(
        """
        UPDATE extension_install_registry
        SET
            is_admin_install = 1,
            admin_reason = ?,
            current_operator_id = CASE
                WHEN ? IS NOT NULL AND TRIM(?) <> '' THEN ?
                ELSE current_operator_id
            END,
            agency_id = CASE
                WHEN ? IS NOT NULL AND TRIM(?) <> '' THEN ?
                ELSE agency_id
            END,
            last_seen_version = CASE
                WHEN ? IS NOT NULL AND TRIM(?) <> '' THEN ?
                ELSE last_seen_version
            END,
            last_seen_at = ?,
            updated_at = ?
        WHERE install_id = ?
        """,
        (
            reason,
            operator_key,
            operator_key,
            operator_key,
            agency_key,
            agency_key,
            agency_key,
            version_key,
            version_key,
            version_key,
            now_ms,
            now_ms,
            install_key,
        ),
    )
    row = fetch_extension_install_registry(conn, install_key)
    if not row:
        raise ValueError("install registry row not found")
    return row


def fetch_operator_install_binding(
    conn: sqlite3.Connection, operator_id: str
) -> Optional[sqlite3.Row]:
    operator_key = str(operator_id or "").strip()
    if not operator_key:
        return None
    return conn.execute(
        """
        SELECT *
        FROM operator_install_binding
        WHERE operator_id = ?
        LIMIT 1
        """,
        (operator_key,),
    ).fetchone()


def close_open_binding_history(
    conn: sqlite3.Connection,
    operator_id: str,
    install_id: str,
    replaced_at: int,
) -> None:
    conn.execute(
        """
        UPDATE operator_install_binding_history
        SET replaced_at = ?
        WHERE operator_id = ?
          AND install_id = ?
          AND replaced_at IS NULL
        """,
        (replaced_at, operator_id, install_id),
    )


def upsert_operator_install_binding(
    conn: sqlite3.Connection,
    operator_id: str,
    install_id: str,
    now_ms: int,
    extension_version: Optional[str] = None,
    agency_id: Optional[str] = None,
    reason: str = "first_bind",
) -> Tuple[sqlite3.Row, bool]:
    operator_key = str(operator_id or "").strip()
    install_key = str(install_id or "").strip()
    agency_key = str(agency_id or "").strip() or None
    version_key = str(extension_version or "").strip() or None
    if not operator_key or not install_key:
        raise ValueError("operator_id and install_id are required")
    existing = fetch_operator_install_binding(conn, operator_key)
    replaced_previous = False
    previous_install_id = None
    if not existing:
        conn.execute(
            """
            INSERT INTO operator_install_binding (
                operator_id,
                active_install_id,
                active_extension_version,
                agency_id,
                bound_at,
                updated_at,
                previous_install_id,
                previous_replaced_at
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                operator_key,
                install_key,
                version_key,
                agency_key,
                now_ms,
                now_ms,
            ),
        )
    elif str(existing["active_install_id"] or "").strip() == install_key:
        conn.execute(
            """
            UPDATE operator_install_binding
            SET
                active_extension_version = CASE
                    WHEN ? IS NOT NULL AND TRIM(?) <> '' THEN ?
                    ELSE active_extension_version
                END,
                agency_id = CASE
                    WHEN ? IS NOT NULL AND TRIM(?) <> '' THEN ?
                    ELSE agency_id
                END,
                updated_at = ?
            WHERE operator_id = ?
            """,
            (
                version_key,
                version_key,
                version_key,
                agency_key,
                agency_key,
                agency_key,
                now_ms,
                operator_key,
            ),
        )
    else:
        replaced_previous = True
        previous_install_id = str(existing["active_install_id"] or "").strip() or None
        if previous_install_id:
            close_open_binding_history(conn, operator_key, previous_install_id, now_ms)
        conn.execute(
            """
            UPDATE operator_install_binding
            SET
                active_install_id = ?,
                active_extension_version = ?,
                agency_id = ?,
                updated_at = ?,
                previous_install_id = ?,
                previous_replaced_at = ?
            WHERE operator_id = ?
            """,
            (
                install_key,
                version_key,
                agency_key,
                now_ms,
                previous_install_id,
                now_ms,
                operator_key,
            ),
        )
    current = fetch_operator_install_binding(conn, operator_key)
    if not current:
        raise ValueError("operator binding row was not created")
    if not existing or replaced_previous:
        conn.execute(
            """
            INSERT INTO operator_install_binding_history (
                operator_id,
                install_id,
                extension_version,
                agency_id,
                bound_at,
                replaced_at,
                reason
            )
            VALUES (?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                operator_key,
                install_key,
                version_key,
                agency_key,
                now_ms,
                reason,
            ),
        )
    return current, replaced_previous


def remove_bindings_for_install(
    conn: sqlite3.Connection, install_id: str, now_ms: int
) -> None:
    install_key = str(install_id or "").strip()
    if not install_key:
        return
    rows = conn.execute(
        """
        SELECT operator_id, active_install_id
        FROM operator_install_binding
        WHERE active_install_id = ?
        """,
        (install_key,),
    ).fetchall()
    for row in rows:
        operator_id = str(row["operator_id"] or "").strip()
        active_install_id = str(row["active_install_id"] or "").strip()
        if operator_id and active_install_id:
            close_open_binding_history(conn, operator_id, active_install_id, now_ms)
    conn.execute(
        """
        DELETE FROM operator_install_binding
        WHERE active_install_id = ?
        """,
        (install_key,),
    )


def fetch_admin_operator_summary_items(
    conn: sqlite3.Connection,
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            eo.operator_id,
            MIN(eo.first_used_at) AS first_used_at,
            MAX(eo.last_used_at) AS last_used_at,
            COUNT(DISTINCT eo.install_id) AS install_count,
            COUNT(DISTINCT eo.password_id) AS password_count,
            (
                SELECT eo2.agency_id
                FROM extension_password_usage_operators eo2
                WHERE eo2.operator_id = eo.operator_id
                  AND eo2.agency_id IS NOT NULL
                  AND TRIM(eo2.agency_id) <> ''
                ORDER BY eo2.last_used_at DESC, eo2.id DESC
                LIMIT 1
            ) AS agency_id,
            COALESCE((
                SELECT aogu.generation_count
                FROM audio_operator_generation_usage aogu
                WHERE aogu.operator_id = eo.operator_id
                LIMIT 1
            ), 0) AS tts_generation_count,
            COALESCE((
                SELECT oib.active_install_id
                FROM operator_install_binding oib
                WHERE oib.operator_id = eo.operator_id
                LIMIT 1
            ), '') AS active_install_id,
            COALESCE((
                SELECT oib.active_extension_version
                FROM operator_install_binding oib
                WHERE oib.operator_id = eo.operator_id
                LIMIT 1
            ), '') AS active_extension_version,
            COALESCE((
                SELECT COUNT(DISTINCT ioh.install_id)
                FROM install_operator_history ioh
                INNER JOIN extension_install_registry eir
                    ON eir.install_id = ioh.install_id
                WHERE ioh.operator_id = eo.operator_id
                  AND eir.is_admin_install = 1
            ), 0) AS admin_install_count
        FROM extension_password_usage_operators eo
        GROUP BY eo.operator_id
        ORDER BY install_count DESC, eo.operator_id ASC
        """
    ).fetchall()
    team_names = fetch_operator_team_names(
        conn,
        [str(row["operator_id"] or "").strip() for row in rows],
    )
    admin_install_meta = fetch_admin_install_name_meta(conn)
    install_admin_flags = fetch_install_admin_flags(
        conn,
        [str(row["active_install_id"] or "").strip() for row in rows],
    )
    admin_names_by_operator = fetch_related_admin_names_by_operator(
        conn,
        [str(row["operator_id"] or "").strip() for row in rows],
        admin_install_meta,
    )
    multi_install_meta_by_operator = fetch_multi_install_current_version_meta(
        conn,
        [str(row["operator_id"] or "").strip() for row in rows],
    )
    items: List[Dict[str, Any]] = []
    for row in rows:
        operator_id = str(row["operator_id"] or "").strip()
        if not operator_id:
            continue
        active_install_id = str(row["active_install_id"] or "").strip()
        active_install_meta = admin_install_meta.get(active_install_id, {})
        multi_install_meta = multi_install_meta_by_operator.get(operator_id, {})
        items.append(
            {
                "operator_id": operator_id,
                "agency_id": str(row["agency_id"] or "").strip(),
                "team_name": team_names.get(operator_id, ""),
                "install_count": int(row["install_count"] or 0),
                "password_count": int(row["password_count"] or 0),
                "tts_generation_count": int(row["tts_generation_count"] or 0),
                "active_install_id": active_install_id,
                "active_install_is_admin": bool(
                    install_admin_flags.get(active_install_id, False)
                ),
                "active_install_admin_name": str(
                    active_install_meta.get("admin_name") or ""
                ).strip(),
                "active_extension_version": str(
                    row["active_extension_version"] or ""
                ).strip(),
                "admin_install_count": int(row["admin_install_count"] or 0),
                "admin_names": admin_names_by_operator.get(operator_id, []),
                "multi_install_current_version": bool(
                    multi_install_meta.get("multi_install_current_version")
                ),
                "multi_install_current_version_count": int(
                    multi_install_meta.get("multi_install_current_version_count") or 0
                ),
                "multi_install_current_version_install_ids": list(
                    multi_install_meta.get("multi_install_current_version_install_ids") or []
                ),
                "multi_install_current_version_version": str(
                    multi_install_meta.get("multi_install_current_version_version") or ""
                ).strip(),
                "first_used_at": int(row["first_used_at"] or 0),
                "last_used_at": int(row["last_used_at"] or 0),
            }
        )
    return items


def fetch_admin_passwords(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    latest_version_key = str(LATEST_EXTENSION_VERSION or "").strip()
    rows = conn.execute(
        """
        SELECT
            p.id,
            p.name,
            p.password,
            p.team_name,
            p.is_active,
            p.created_at,
            p.updated_at,
            COALESCE(COUNT(u.id), 0) AS unique_users,
            0 AS total_success,
            MAX(u.last_used_at) AS last_used_at
        FROM extension_passwords p
        LEFT JOIN extension_password_usages u
            ON u.password_id = p.id
        WHERE p.deleted_at IS NULL
        GROUP BY p.id
        ORDER BY p.created_at DESC, p.id DESC
        """
    ).fetchall()
    operators_by_password: Dict[int, List[Dict[str, Any]]] = {}
    install_groups_by_password: Dict[int, List[Dict[str, Any]]] = {}
    global_operator_summary_by_id = {
        str(item.get("operator_id") or "").strip(): item
        for item in fetch_admin_operator_summary_items(conn)
    }
    operator_rows = conn.execute(
        """
        SELECT
            password_id,
            operator_id,
            MIN(first_used_at) AS first_used_at,
            MAX(last_used_at) AS last_used_at,
            0 AS success_count,
            COUNT(DISTINCT install_id) AS install_count,
            (
                SELECT eo2.agency_id
                FROM extension_password_usage_operators eo2
                WHERE eo2.password_id = eo.password_id
                  AND eo2.operator_id = eo.operator_id
                  AND eo2.agency_id IS NOT NULL
                  AND TRIM(eo2.agency_id) <> ''
                ORDER BY eo2.last_used_at DESC, eo2.id DESC
                LIMIT 1
            ) AS agency_id
        FROM extension_password_usage_operators eo
        GROUP BY password_id, operator_id
        ORDER BY MAX(last_used_at) DESC, operator_id ASC
        """
    ).fetchall()
    for row in operator_rows:
        password_id = int(row["password_id"] or 0)
        if password_id <= 0:
            continue
        operator_id = str(row["operator_id"] or "").strip()
        global_entry = global_operator_summary_by_id.get(operator_id, {})
        active_extension_version = str(
            global_entry.get("active_extension_version") or ""
        ).strip()
        if latest_version_key and active_extension_version != latest_version_key:
            continue
        operators_by_password.setdefault(password_id, []).append(
            {
                "operator_id": operator_id,
                "agency_id": str(
                    row["agency_id"] or global_entry.get("agency_id") or ""
                ).strip(),
                "team_name": str(global_entry.get("team_name") or "").strip(),
                "first_used_at": int(row["first_used_at"] or 0),
                "last_used_at": int(row["last_used_at"] or 0),
                "success_count": int(row["success_count"] or 0),
                "install_count": int(row["install_count"] or 0),
                "global_install_count": int(global_entry.get("install_count") or 0),
                "password_count": int(global_entry.get("password_count") or 0),
                "tts_generation_count": int(
                    global_entry.get("tts_generation_count") or 0
                ),
                "admin_install_count": int(
                    global_entry.get("admin_install_count") or 0
                ),
                "active_install_id": str(
                    global_entry.get("active_install_id") or ""
                ).strip(),
                "active_install_is_admin": bool(
                    global_entry.get("active_install_is_admin")
                ),
                "active_install_admin_name": str(
                    global_entry.get("active_install_admin_name") or ""
                ).strip(),
                "active_extension_version": active_extension_version,
                "admin_names": list(global_entry.get("admin_names") or []),
                "multi_install_current_version": bool(
                    global_entry.get("multi_install_current_version")
                ),
                "multi_install_current_version_count": int(
                    global_entry.get("multi_install_current_version_count") or 0
                ),
                "multi_install_current_version_install_ids": list(
                    global_entry.get("multi_install_current_version_install_ids") or []
                ),
                "multi_install_current_version_version": str(
                    global_entry.get("multi_install_current_version_version") or ""
                ).strip(),
            }
        )
    install_group_rows = conn.execute(
        """
        SELECT
            u.password_id,
            u.install_id,
            u.extension_version,
            u.first_used_at,
            u.last_used_at,
            COUNT(DISTINCT eo.operator_id) AS operators_count,
            (
                SELECT GROUP_CONCAT(sorted_operators.operator_id)
                FROM (
                    SELECT DISTINCT eo2.operator_id AS operator_id
                    FROM extension_password_usage_operators eo2
                    WHERE eo2.password_id = u.password_id
                      AND eo2.install_id = u.install_id
                    ORDER BY eo2.operator_id ASC
                ) AS sorted_operators
            ) AS operators
        FROM extension_password_usages u
        LEFT JOIN extension_password_usage_operators eo
            ON eo.password_id = u.password_id
           AND eo.install_id = u.install_id
        GROUP BY u.password_id, u.install_id
        ORDER BY operators_count DESC, u.install_id ASC
        """
    ).fetchall()
    for row in install_group_rows:
        password_id = int(row["password_id"] or 0)
        if password_id <= 0:
            continue
        install_id = str(row["install_id"] or "").strip()
        if not install_id:
            continue
        operators = [
            operator_id.strip()
            for operator_id in str(row["operators"] or "").split(",")
            if operator_id and operator_id.strip()
        ]
        install_groups_by_password.setdefault(password_id, []).append(
            {
                "install_id": install_id,
                "install_short": install_id[:8],
                "extension_version": str(row["extension_version"] or "").strip(),
                "operators_count": int(row["operators_count"] or 0),
                "operators": operators,
                "first_used_at": int(row["first_used_at"] or 0),
                "last_used_at": int(row["last_used_at"] or 0),
            }
        )
    install_group_admin_flags = fetch_install_admin_flags(
        conn,
        [
            str(entry.get("install_id") or "").strip()
            for items in install_groups_by_password.values()
            for entry in items
        ],
    )
    admin_install_meta = fetch_admin_install_name_meta(conn)
    for items in install_groups_by_password.values():
        for entry in items:
            install_id = str(entry.get("install_id") or "").strip()
            entry["is_admin_install"] = bool(
                install_group_admin_flags.get(install_id, False)
            )
            entry["admin_name"] = str(
                admin_install_meta.get(install_id, {}).get("admin_name") or ""
            ).strip()
    if latest_version_key:
        active_install_ids_by_password: Dict[int, Set[str]] = {}
        for password_id, operator_items in operators_by_password.items():
            active_ids = {
                str(entry.get("active_install_id") or "").strip()
                for entry in operator_items
                if str(entry.get("active_install_id") or "").strip()
            }
            active_install_ids_by_password[password_id] = active_ids
        filtered_install_groups_by_password: Dict[int, List[Dict[str, Any]]] = {}
        for password_id, items in install_groups_by_password.items():
            active_install_ids = active_install_ids_by_password.get(password_id, set())
            if not active_install_ids:
                filtered_install_groups_by_password[password_id] = []
                continue
            filtered_install_groups_by_password[password_id] = [
                entry
                for entry in items
                if str(entry.get("install_id") or "").strip() in active_install_ids
            ]
        install_groups_by_password = filtered_install_groups_by_password
    out: List[Dict[str, Any]] = []
    for row in rows:
        password_id = int(row["id"])
        password_operators = operators_by_password.get(password_id, [])
        out.append(
            {
                "id": password_id,
                "name": str(row["name"] or ""),
                "password": str(row["password"] or ""),
                "team_name": str(row["team_name"] or "").strip(),
                "is_active": bool(row["is_active"]),
                "created_at": int(row["created_at"] or 0),
                "updated_at": int(row["updated_at"] or 0),
                "unique_users": len(password_operators)
                if latest_version_key
                else int(row["unique_users"] or 0),
                "total_success": int(row["total_success"] or 0),
                "last_used_at": int(row["last_used_at"] or 0)
                if row["last_used_at"] is not None
                else None,
                "operators": password_operators,
                "install_groups": install_groups_by_password.get(password_id, []),
            }
        )
    return out


def fetch_admin_operators_summary(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    return fetch_admin_operator_summary_items(conn)


def fetch_admin_admins_summary(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    items = fetch_admin_install_summary_items(conn)
    latest_version_key = str(LATEST_EXTENSION_VERSION or "").strip()
    if not latest_version_key:
        return items
    return [
        item
        for item in items
        if str(item.get("install_version") or "").strip() == latest_version_key
    ]


def fetch_admin_agencies_summary(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            eo.agency_id,
            MIN(eo.first_used_at) AS first_used_at,
            MAX(eo.last_used_at) AS last_used_at,
            COUNT(DISTINCT eo.install_id) AS install_count,
            COUNT(DISTINCT eo.operator_id) AS operator_count,
            COUNT(DISTINCT eo.password_id) AS password_count,
            COALESCE((
                SELECT SUM(aogu.generation_count)
                FROM audio_operator_generation_usage aogu
                WHERE aogu.agency_id = eo.agency_id
            ), 0) AS tts_generation_count,
            COALESCE((
                SELECT COUNT(*)
                FROM operator_install_binding oib
                WHERE oib.agency_id = eo.agency_id
            ), 0) AS active_operator_install_count,
            COALESCE((
                SELECT COUNT(DISTINCT ioh.install_id)
                FROM install_operator_history ioh
                INNER JOIN extension_install_registry eir
                    ON eir.install_id = ioh.install_id
                WHERE ioh.agency_id = eo.agency_id
                  AND eir.is_admin_install = 1
            ), 0) AS admin_install_count
        FROM extension_password_usage_operators eo
        WHERE eo.agency_id IS NOT NULL
          AND TRIM(eo.agency_id) <> ''
        GROUP BY eo.agency_id
        ORDER BY install_count DESC, eo.agency_id ASC
        """
    ).fetchall()
    admin_install_meta = fetch_admin_install_name_meta(conn)
    admin_names_by_agency = fetch_related_admin_names_by_agency(
        conn,
        [str(row["agency_id"] or "").strip() for row in rows],
        admin_install_meta,
    )
    items: List[Dict[str, Any]] = []
    for row in rows:
        agency_id = str(row["agency_id"] or "").strip()
        if not agency_id:
            continue
        items.append(
            {
                "agency_id": agency_id,
                "install_count": int(row["install_count"] or 0),
                "operator_count": int(row["operator_count"] or 0),
                "password_count": int(row["password_count"] or 0),
                "tts_generation_count": int(row["tts_generation_count"] or 0),
                "active_operator_install_count": int(
                    row["active_operator_install_count"] or 0
                ),
                "admin_install_count": int(row["admin_install_count"] or 0),
                "admin_names": admin_names_by_agency.get(agency_id, []),
                "first_used_at": int(row["first_used_at"] or 0),
                "last_used_at": int(row["last_used_at"] or 0),
            }
        )
    return items


def fetch_admin_agency_details(
    conn: sqlite3.Connection, agency_id: str
) -> Optional[Dict[str, Any]]:
    agency_key = str(agency_id or "").strip()
    if not agency_key:
        return None
    summary_row = conn.execute(
        """
        SELECT
            eo.agency_id,
            MIN(eo.first_used_at) AS first_used_at,
            MAX(eo.last_used_at) AS last_used_at,
            COUNT(DISTINCT eo.install_id) AS install_count,
            COUNT(DISTINCT eo.operator_id) AS operator_count,
            COUNT(DISTINCT eo.password_id) AS password_count,
            COALESCE((
                SELECT SUM(aogu.generation_count)
                FROM audio_operator_generation_usage aogu
                WHERE aogu.agency_id = eo.agency_id
            ), 0) AS tts_generation_count,
            COALESCE((
                SELECT COUNT(*)
                FROM operator_install_binding oib
                WHERE oib.agency_id = eo.agency_id
            ), 0) AS active_operator_install_count,
            COALESCE((
                SELECT COUNT(DISTINCT ioh.install_id)
                FROM install_operator_history ioh
                INNER JOIN extension_install_registry eir
                    ON eir.install_id = ioh.install_id
                WHERE ioh.agency_id = eo.agency_id
                  AND eir.is_admin_install = 1
            ), 0) AS admin_install_count
        FROM extension_password_usage_operators eo
        WHERE eo.agency_id = ?
        GROUP BY eo.agency_id
        """,
        (agency_key,),
    ).fetchone()
    if not summary_row:
        return None

    operator_rows = conn.execute(
        """
        SELECT
            eo.operator_id,
            MIN(eo.first_used_at) AS first_used_at,
            MAX(eo.last_used_at) AS last_used_at,
            COUNT(DISTINCT eo.install_id) AS install_count,
            COUNT(DISTINCT eo.password_id) AS password_count,
            COALESCE((
                SELECT aogu.generation_count
                FROM audio_operator_generation_usage aogu
                WHERE aogu.operator_id = eo.operator_id
                  AND aogu.agency_id = ?
                LIMIT 1
            ), 0) AS tts_generation_count,
            COALESCE((
                SELECT oib.active_install_id
                FROM operator_install_binding oib
                WHERE oib.operator_id = eo.operator_id
                LIMIT 1
            ), '') AS active_install_id,
            COALESCE((
                SELECT oib.active_extension_version
                FROM operator_install_binding oib
                WHERE oib.operator_id = eo.operator_id
                LIMIT 1
            ), '') AS active_extension_version,
            COALESCE((
                SELECT COUNT(DISTINCT ioh.install_id)
                FROM install_operator_history ioh
                INNER JOIN extension_install_registry eir
                    ON eir.install_id = ioh.install_id
                WHERE ioh.operator_id = eo.operator_id
                  AND eir.is_admin_install = 1
            ), 0) AS admin_install_count
        FROM extension_password_usage_operators eo
        WHERE eo.agency_id = ?
        GROUP BY eo.operator_id
        ORDER BY MAX(eo.last_used_at) DESC, eo.operator_id ASC
        """,
        (agency_key, agency_key),
    ).fetchall()
    team_names_by_operator = fetch_operator_team_names(
        conn,
        [str(row["operator_id"] or "").strip() for row in operator_rows],
    )
    admin_install_meta = fetch_admin_install_name_meta(conn)
    operator_active_install_admin_flags = fetch_install_admin_flags(
        conn,
        [str(row["active_install_id"] or "").strip() for row in operator_rows],
    )
    admin_names_by_operator = fetch_related_admin_names_by_operator(
        conn,
        [str(row["operator_id"] or "").strip() for row in operator_rows],
        admin_install_meta,
    )
    multi_install_meta_by_operator = fetch_multi_install_current_version_meta(
        conn,
        [str(row["operator_id"] or "").strip() for row in operator_rows],
    )
    operators: List[Dict[str, Any]] = []
    for row in operator_rows:
        operator_id = str(row["operator_id"] or "").strip()
        if not operator_id:
            continue
        active_install_id = str(row["active_install_id"] or "").strip()
        active_install_meta = admin_install_meta.get(active_install_id, {})
        multi_install_meta = multi_install_meta_by_operator.get(operator_id, {})
        operators.append(
            {
                "operator_id": operator_id,
                "agency_id": agency_key,
                "team_name": team_names_by_operator.get(operator_id, ""),
                "install_count": int(row["install_count"] or 0),
                "password_count": int(row["password_count"] or 0),
                "tts_generation_count": int(row["tts_generation_count"] or 0),
                "active_install_id": active_install_id,
                "active_install_is_admin": bool(
                    operator_active_install_admin_flags.get(active_install_id, False)
                ),
                "active_install_admin_name": str(
                    active_install_meta.get("admin_name") or ""
                ).strip(),
                "active_extension_version": str(
                    row["active_extension_version"] or ""
                ).strip(),
                "admin_install_count": int(row["admin_install_count"] or 0),
                "admin_names": admin_names_by_operator.get(operator_id, []),
                "multi_install_current_version": bool(
                    multi_install_meta.get("multi_install_current_version")
                ),
                "multi_install_current_version_count": int(
                    multi_install_meta.get("multi_install_current_version_count") or 0
                ),
                "multi_install_current_version_install_ids": list(
                    multi_install_meta.get("multi_install_current_version_install_ids") or []
                ),
                "multi_install_current_version_version": str(
                    multi_install_meta.get("multi_install_current_version_version") or ""
                ).strip(),
                "first_used_at": int(row["first_used_at"] or 0),
                "last_used_at": int(row["last_used_at"] or 0),
            }
        )

    install_rows = conn.execute(
        """
        SELECT
            oib.active_install_id AS install_id,
            MIN(oib.bound_at) AS first_used_at,
            MAX(oib.updated_at) AS last_used_at,
            COUNT(DISTINCT oib.operator_id) AS operators_count,
            COALESCE((
                SELECT COUNT(DISTINCT eo2.password_id)
                FROM extension_password_usage_operators eo2
                WHERE eo2.agency_id = ?
                  AND eo2.install_id = oib.active_install_id
            ), 0) AS password_count,
            (
                CASE
                    WHEN oib.active_extension_version IS NOT NULL
                         AND TRIM(oib.active_extension_version) <> ''
                    THEN oib.active_extension_version
                    ELSE COALESCE((
                        SELECT u2.extension_version
                        FROM extension_password_usages u2
                        INNER JOIN extension_password_usage_operators eo2
                            ON eo2.password_id = u2.password_id
                           AND eo2.install_id = u2.install_id
                        WHERE eo2.agency_id = ?
                          AND eo2.install_id = oib.active_install_id
                        ORDER BY u2.last_used_at DESC, u2.id DESC
                        LIMIT 1
                    ), '')
                END
            ) AS extension_version,
            (
                SELECT GROUP_CONCAT(sorted_operators.operator_id)
                FROM (
                    SELECT DISTINCT oib2.operator_id AS operator_id
                    FROM operator_install_binding oib2
                    WHERE oib2.agency_id = ?
                      AND oib2.active_install_id = oib.active_install_id
                    ORDER BY oib2.operator_id ASC
                ) AS sorted_operators
            ) AS operators,
            (
                SELECT GROUP_CONCAT(sorted_teams.team_name)
                FROM (
                    SELECT DISTINCT TRIM(p.team_name) AS team_name
                    FROM extension_password_usage_operators eo4
                    INNER JOIN extension_passwords p
                        ON p.id = eo4.password_id
                    WHERE eo4.agency_id = ?
                      AND eo4.install_id = oib.active_install_id
                      AND p.team_name IS NOT NULL
                      AND TRIM(p.team_name) <> ''
                    ORDER BY team_name ASC
                ) AS sorted_teams
            ) AS install_team_names,
            (
                SELECT GROUP_CONCAT(sorted_passwords.password_name)
                FROM (
                    SELECT DISTINCT TRIM(p2.name) AS password_name
                    FROM extension_password_usage_operators eo5
                    INNER JOIN extension_passwords p2
                        ON p2.id = eo5.password_id
                    WHERE eo5.agency_id = ?
                      AND eo5.install_id = oib.active_install_id
                      AND p2.name IS NOT NULL
                      AND TRIM(p2.name) <> ''
                    ORDER BY password_name ASC
                ) AS sorted_passwords
            ) AS password_names
        FROM operator_install_binding oib
        WHERE oib.agency_id = ?
          AND oib.active_install_id IS NOT NULL
          AND TRIM(oib.active_install_id) <> ''
          AND NOT EXISTS (
            SELECT 1
            FROM extension_install_registry eir
            WHERE eir.install_id = oib.active_install_id
              AND eir.is_admin_install = 1
          )
        GROUP BY oib.active_install_id
        ORDER BY MAX(oib.updated_at) DESC, oib.active_install_id ASC
        """,
        (
            agency_key,
            agency_key,
            agency_key,
            agency_key,
            agency_key,
            agency_key,
        ),
    ).fetchall()
    install_groups: List[Dict[str, Any]] = []
    install_group_admin_flags = fetch_install_admin_flags(
        conn,
        [str(row["install_id"] or "").strip() for row in install_rows],
    )
    for row in install_rows:
        install_id = str(row["install_id"] or "").strip()
        if not install_id:
            continue
        operators = [
            operator_id.strip()
            for operator_id in str(row["operators"] or "").split(",")
            if operator_id and operator_id.strip()
        ]
        install_groups.append(
            {
                "install_id": install_id,
                "install_short": install_id[:8],
                "extension_version": str(row["extension_version"] or "").strip(),
                "operators_count": int(row["operators_count"] or 0),
                "password_count": int(row["password_count"] or 0),
                "operators": operators,
                "operator_teams": ", ".join(
                    sorted(
                        {
                            team_name
                            for team_name in (
                                team_names_by_operator.get(operator_id, "")
                                for operator_id in operators
                            )
                            if team_name
                        }
                    )
                ),
                "install_team_names": str(row["install_team_names"] or "").strip(),
                "password_names": str(row["password_names"] or "").strip(),
                "first_used_at": int(row["first_used_at"] or 0),
                "last_used_at": int(row["last_used_at"] or 0),
                "is_admin_install": bool(
                    install_group_admin_flags.get(install_id, False)
                ),
                "admin_name": str(
                    admin_install_meta.get(install_id, {}).get("admin_name") or ""
                ).strip(),
            }
        )
    admin_installs = fetch_admin_install_summary_items(conn, agency_key)

    return {
        "agency_id": agency_key,
        "install_count": int(summary_row["install_count"] or 0),
        "operator_count": int(summary_row["operator_count"] or 0),
        "password_count": int(summary_row["password_count"] or 0),
        "tts_generation_count": int(summary_row["tts_generation_count"] or 0),
        "active_operator_install_count": int(
            summary_row["active_operator_install_count"] or 0
        ),
        "admin_install_count": int(summary_row["admin_install_count"] or 0),
        "first_used_at": int(summary_row["first_used_at"] or 0),
        "last_used_at": int(summary_row["last_used_at"] or 0),
        "operators": operators,
        "install_groups": install_groups,
        "admin_installs": admin_installs,
    }


@app.get("/api/admin/passwords")
def admin_list_passwords(_=Depends(admin_auth)):
    conn = get_conn()
    try:
        items = fetch_admin_passwords(conn)
        return {"ok": True, "items": items}
    finally:
        conn.close()


@app.get("/api/admin/operators/summary")
def admin_list_operators_summary(_=Depends(admin_auth)):
    conn = get_conn()
    try:
        items = fetch_admin_operators_summary(conn)
        return {"ok": True, "items": items}
    finally:
        conn.close()


@app.get("/api/admin/admins/summary")
def admin_list_admins_summary(_=Depends(admin_auth)):
    conn = get_conn()
    try:
        items = fetch_admin_admins_summary(conn)
        return {"ok": True, "items": items}
    finally:
        conn.close()


@app.get("/api/admin/agencies/summary")
def admin_list_agencies_summary(_=Depends(admin_auth)):
    conn = get_conn()
    try:
        items = fetch_admin_agencies_summary(conn)
        return {"ok": True, "items": items}
    finally:
        conn.close()


@app.get("/api/admin/agencies/details")
def admin_get_agency_details(agency_id: str, _=Depends(admin_auth)):
    conn = get_conn()
    try:
        item = fetch_admin_agency_details(conn, agency_id)
        if not item:
            raise HTTPException(status_code=404, detail="Agency not found")
        return {"ok": True, "item": item}
    finally:
        conn.close()


@app.post("/api/admin/passwords")
def admin_create_password(payload: AdminPasswordCreatePayload, _=Depends(admin_auth)):
    now_ms = int(time.time() * 1000)
    name = payload.name.strip()
    password = payload.password.strip()
    team_name = (payload.team_name or "").strip() or None
    conn = get_conn()
    try:
        with conn:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO extension_passwords (
                        name,
                        password,
                        team_name,
                        is_active,
                        created_at,
                        updated_at,
                        deleted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        name,
                        password,
                        team_name,
                        1 if payload.is_active else 0,
                        now_ms,
                        now_ms,
                    ),
                )
            except sqlite3.IntegrityError:
                raise HTTPException(
                    status_code=409, detail="Password name already exists"
                )
        return {"ok": True, "id": int(cur.lastrowid or 0)}
    finally:
        conn.close()


@app.patch("/api/admin/passwords/{password_id}")
def admin_update_password(
    password_id: int, payload: AdminPasswordUpdatePayload, _=Depends(admin_auth)
):
    if password_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid password id")
    updates: List[str] = []
    params: List[Any] = []
    if payload.name is not None:
        updates.append("name = ?")
        params.append(payload.name.strip())
    if payload.password is not None:
        updates.append("password = ?")
        params.append(payload.password.strip())
    if payload.team_name is not None:
        updates.append("team_name = ?")
        params.append(payload.team_name.strip() or None)
    if payload.is_active is not None:
        updates.append("is_active = ?")
        params.append(1 if payload.is_active else 0)
    if not updates:
        return {"ok": True, "updated": 0}
    updates.append("updated_at = ?")
    params.append(int(time.time() * 1000))
    params.append(password_id)
    conn = get_conn()
    try:
        with conn:
            try:
                cur = conn.execute(
                    f"""
                    UPDATE extension_passwords
                    SET {", ".join(updates)}
                    WHERE id = ? AND deleted_at IS NULL
                    """,
                    tuple(params),
                )
            except sqlite3.IntegrityError:
                raise HTTPException(
                    status_code=409, detail="Password name already exists"
                )
        if not cur.rowcount:
            raise HTTPException(status_code=404, detail="Password not found")
        return {"ok": True, "updated": int(cur.rowcount)}
    finally:
        conn.close()


@app.delete("/api/admin/passwords/{password_id}")
def admin_delete_password(password_id: int, _=Depends(admin_auth)):
    if password_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid password id")
    now_ms = int(time.time() * 1000)
    conn = get_conn()
    try:
        with conn:
            cur = conn.execute(
                """
                UPDATE extension_passwords
                SET deleted_at = ?, updated_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (now_ms, now_ms, password_id),
            )
        if not cur.rowcount:
            raise HTTPException(status_code=404, detail="Password not found")
        return {"ok": True, "deleted": int(cur.rowcount)}
    finally:
        conn.close()


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/count")
def count(male_id: str, _=Depends(auth)):
    """
    Совместимость со старым API:
    возвращает количество отчётов в таблице reports для указанного male_id.
    """
    if not TEN_DIGITS.match(male_id):
        raise HTTPException(status_code=400, detail="male_id must be exactly 10 digits")
    n = get_count_from_db(male_id)
    return {"ok": True, "male_id": male_id, "count": n}


@app.post("/api/operators/top/sync")
def sync_global_top(payload: TopSyncPayload, _=Depends(auth)):
    if not payload.operators:
        return {"ok": True, "updated": 0}
    conn = get_conn()
    try:
        updated = 0
        with conn:
            for item in payload.operators:
                try:
                    changed_daily = upsert_top_entry(conn, item)
                    changed_record = upsert_top_record_entry(conn, item)
                    if changed_daily or changed_record:
                        updated += 1
                except Exception:
                    continue
        return {"ok": True, "updated": updated}
    finally:
        conn.close()


@app.get("/api/operators/top")
def get_global_top(day_key: Optional[str] = None, _=Depends(auth)):
    target_day = normalize_top_day_key(day_key)
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT operator_id, operator_name, shift_balance, hour_balance, updated_at, day_key
            FROM operators_top
            WHERE day_key = ?
            ORDER BY shift_balance DESC, hour_balance DESC, operator_id ASC
            """,
            (target_day,),
        )
        items = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "day_key": target_day, "operators": items}
    finally:
        conn.close()


@app.get("/api/operators/top/records")
def get_global_top_records(limit: int = 250, _=Depends(auth)):
    capped_limit = max(1, min(int(limit or 250), 250))
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT operator_id,
                   operator_name,
                   record_balance AS shift_balance,
                   updated_at
            FROM operators_top_records
            ORDER BY record_balance DESC, updated_at DESC, operator_id ASC
            LIMIT ?
            """,
            (capped_limit,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "operators": rows, "limit": capped_limit}
    finally:
        conn.close()


@app.post("/api/operators/actions/sync")
def sync_global_actions(payload: TopActionsSyncPayload, _=Depends(auth)):
    if not payload.operators:
        return {"ok": True, "updated": 0}
    conn = get_conn()
    try:
        updated = 0
        with conn:
            for item in payload.operators:
                try:
                    changed_daily = upsert_actions_entry(conn, item)
                    changed_record = upsert_action_record_entry(conn, item)
                    if changed_daily or changed_record:
                        updated += 1
                except Exception:
                    continue
        return {"ok": True, "updated": updated}
    finally:
        conn.close()


@app.get("/api/operators/actions")
def get_global_actions(day_key: Optional[str] = None, _=Depends(auth)):
    target_day = normalize_top_day_key(day_key)
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT operator_id,
                   operator_name,
                   shift_actions,
                   hour_actions,
                   updated_at,
                   day_key
            FROM operators_actions_top
            WHERE day_key = ?
            ORDER BY shift_actions DESC, hour_actions DESC, operator_id ASC
            """,
            (target_day,),
        )
        items = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "day_key": target_day, "operators": items}
    finally:
        conn.close()


@app.get("/api/operators/actions/records")
def get_global_action_records(limit: int = 250, _=Depends(auth)):
    capped_limit = max(1, min(int(limit or 250), 250))
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT operator_id,
                   operator_name,
                   record_actions AS shift_actions,
                   updated_at
            FROM operators_actions_records
            ORDER BY record_actions DESC, updated_at DESC, operator_id ASC
            LIMIT ?
            """,
            (capped_limit,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "operators": rows, "limit": capped_limit}
    finally:
        conn.close()


@app.get("/api/operators/rating")
def get_operators_rating(
    metric: str,
    scope: str,
    day_key: Optional[str] = None,
    limit: int = 50,
    _=Depends(auth),
):
    normalized_metric = (metric or "").strip().lower()
    if normalized_metric not in {"balance", "actions"}:
        raise HTTPException(status_code=400, detail="metric must be balance|actions")

    normalized_scope = (scope or "").strip().lower()
    if normalized_scope not in {"shift", "all_time"}:
        raise HTTPException(status_code=400, detail="scope must be shift|all_time")

    capped_limit = max(1, min(int(limit or 50), 250))
    target_day = (
        normalize_operator_shift_day_key(day_key)
        if normalized_scope == "shift"
        else None
    )

    conn = get_conn()
    try:
        if normalized_scope == "shift":
            cur = conn.execute(
                """
                SELECT
                    operator_id,
                    COALESCE(NULLIF(MAX(operator_name), ''), '') AS operator_name,
                    MAX(balance_total) AS balance_total,
                    MAX(actions_total) AS actions_total,
                    MAX(chat_count) AS chat_count,
                    MAX(mail_count) AS mail_count,
                    MAX(updated_at) AS updated_at
                FROM operator_shift_summary
                WHERE day_key = ?
                GROUP BY operator_id
                """,
                (target_day,),
            )
        else:
            metric_column = "balance_total" if normalized_metric == "balance" else "actions_total"
            cur = conn.execute(
                f"""
                SELECT
                    operator_id,
                    COALESCE(operator_name, '') AS operator_name,
                    balance_total,
                    actions_total,
                    chat_count,
                    mail_count,
                    updated_at
                FROM (
                    SELECT
                        operator_id,
                        operator_name,
                        balance_total,
                        actions_total,
                        chat_count,
                        mail_count,
                        updated_at,
                        day_key,
                        ROW_NUMBER() OVER (
                            PARTITION BY operator_id
                            ORDER BY {metric_column} DESC, updated_at DESC, day_key DESC, operator_id ASC
                        ) AS rn
                    FROM operator_shift_summary
                )
                WHERE rn = 1
                """
            )

        rows = [dict(row) for row in cur.fetchall()]
        team_names_by_operator = fetch_operator_team_names(
            conn,
            [str(row.get("operator_id") or "").strip() for row in rows],
        )
        items: List[Dict[str, Any]] = []
        for row in rows:
            operator_id = str(row.get("operator_id") or "").strip()
            if not operator_id:
                continue
            operator_name = str(row.get("operator_name") or "").strip()
            balance_total = safe_float(row.get("balance_total"))
            actions_total = safe_int(row.get("actions_total"))
            chat_count = safe_int(row.get("chat_count"))
            mail_count = safe_int(row.get("mail_count"))
            updated_at = safe_int(row.get("updated_at"))
            value = balance_total if normalized_metric == "balance" else actions_total
            items.append(
                {
                    "operator_id": operator_id,
                    "operator_name": operator_name,
                    "value": value,
                    "actions_total": actions_total,
                    "chat_count": chat_count,
                    "mail_count": mail_count,
                    "updated_at": updated_at,
                    "balance_total": balance_total,
                    "team_name": team_names_by_operator.get(operator_id, ""),
                }
            )

        def _sort_key(item: Dict[str, Any]):
            value = safe_float(item.get("value"))
            if normalized_metric == "actions":
                second = safe_int(item.get("actions_total"))
            else:
                second = safe_float(item.get("balance_total"))
            operator_id_raw = str(item.get("operator_id") or "").strip()
            operator_id_num = safe_int(operator_id_raw)
            return (-value, -second, operator_id_num, operator_id_raw)

        items.sort(key=_sort_key)
        updated_at_max = max((safe_int(item.get("updated_at")) for item in items), default=0)
        items = items[:capped_limit]

        for item in items:
            item.pop("updated_at", None)
            item.pop("balance_total", None)

        return {
            "ok": True,
            "metric": normalized_metric,
            "scope": normalized_scope,
            "day_key": target_day,
            "updated_at": updated_at_max,
            "items": items,
            "limit": capped_limit,
        }
    finally:
        conn.close()


@app.get("/api/teams/rating")
def get_teams_rating(
    scope: str,
    day_key: Optional[str] = None,
    limit: int = 50,
    _=Depends(auth),
):
    normalized_scope = (scope or "").strip().lower()
    if normalized_scope not in {"shift", "all_time"}:
        raise HTTPException(status_code=400, detail="scope must be shift|all_time")

    capped_limit = max(1, min(int(limit or 50), 250))
    target_day = (
        normalize_operator_shift_day_key(day_key)
        if normalized_scope == "shift"
        else None
    )

    conn = get_conn()
    try:
        if normalized_scope == "shift":
            cur = conn.execute(
                """
                SELECT
                    ots.team_name AS team_name,
                    SUM(COALESCE(oss.balance_total, 0)) AS balance_total,
                    SUM(COALESCE(oss.actions_total, 0)) AS actions_total,
                    SUM(COALESCE(oss.chat_count, 0)) AS chat_count,
                    SUM(COALESCE(oss.mail_count, 0)) AS mail_count,
                    COUNT(DISTINCT oss.operator_id) AS operator_count,
                    MAX(COALESCE(oss.updated_at, 0)) AS updated_at
                FROM operator_shift_summary oss
                INNER JOIN operator_team_state ots
                    ON ots.operator_id = oss.operator_id
                WHERE oss.day_key = ?
                  AND TRIM(COALESCE(ots.team_name, '')) <> ''
                GROUP BY ots.team_name
                """,
                (target_day,),
            )
        else:
            cur = conn.execute(
                """
                SELECT
                    ots.team_name AS team_name,
                    SUM(COALESCE(oss.balance_total, 0)) AS balance_total,
                    SUM(COALESCE(oss.actions_total, 0)) AS actions_total,
                    SUM(COALESCE(oss.chat_count, 0)) AS chat_count,
                    SUM(COALESCE(oss.mail_count, 0)) AS mail_count,
                    COUNT(DISTINCT oss.operator_id) AS operator_count,
                    MAX(COALESCE(oss.updated_at, 0)) AS updated_at
                FROM operator_shift_summary oss
                INNER JOIN operator_team_state ots
                    ON ots.operator_id = oss.operator_id
                WHERE TRIM(COALESCE(ots.team_name, '')) <> ''
                GROUP BY ots.team_name
                """,
            )

        rows = [dict(row) for row in cur.fetchall()]
        items: List[Dict[str, Any]] = []
        for row in rows:
            team_name = str(row.get("team_name") or "").strip()
            if not team_name:
                continue
            balance_total = safe_float(row.get("balance_total"))
            actions_total = safe_int(row.get("actions_total"))
            chat_count = safe_int(row.get("chat_count"))
            mail_count = safe_int(row.get("mail_count"))
            operator_count = safe_int(row.get("operator_count"))
            updated_at = safe_int(row.get("updated_at"))
            items.append(
                {
                    "team_name": team_name,
                    "balance_total": balance_total,
                    "actions_total": actions_total,
                    "chat_count": chat_count,
                    "mail_count": mail_count,
                    "operator_count": operator_count,
                    "updated_at": updated_at,
                }
            )

        def _sort_key(item: Dict[str, Any]):
            balance_total = safe_float(item.get("balance_total"))
            actions_total = safe_int(item.get("actions_total"))
            team_name = str(item.get("team_name") or "").strip().lower()
            return (-balance_total, -actions_total, team_name)

        items.sort(key=_sort_key)
        updated_at_max = max((safe_int(item.get("updated_at")) for item in items), default=0)
        items = items[:capped_limit]
        for item in items:
            item.pop("updated_at", None)

        return {
            "ok": True,
            "scope": normalized_scope,
            "day_key": target_day,
            "updated_at": updated_at_max,
            "items": items,
            "limit": capped_limit,
        }
    finally:
        conn.close()


def upsert_report(conn: sqlite3.Connection, payload: ReportPayload) -> bool:
    updated_at = int(payload.updated_at or 0)
    if updated_at <= 0:
        updated_at = int(time.time() * 1000)
    raw_shift_key = (payload.shift_key or "").strip()
    shift_key = raw_shift_key or compute_shift_key(updated_at)
    fields = {
        "male_id": payload.male_id,
        "female_id": payload.female_id.strip(),
        "operator_id": payload.operator_id.strip(),
        "operator_name": (payload.operator_name or "").strip() or None,
        "shift_key": shift_key,
        "man_name": (payload.man.name or "").strip() or None,
        "man_age": (payload.man.age or "").strip() or None,
        "man_city": (payload.man.city or "").strip() or None,
        "woman_name": (payload.woman.name or "").strip() or None,
        "woman_age": (payload.woman.age or "").strip() or None,
        "woman_city": (payload.woman.city or "").strip() or None,
        "text": payload.text or "",
        "updated_at": updated_at,
        "actions_total": int(payload.actions_total or 0),
        "actions_paid": int(payload.actions_paid or 0),
        "balance_earned": float(payload.balance_earned or 0),
    }
    cur = conn.execute(
        """
        INSERT INTO reports (
            male_id, female_id, operator_id, operator_name, shift_key,
            man_name, man_age, man_city,
            woman_name, woman_age, woman_city,
            text, updated_at, actions_total, actions_paid, balance_earned
        )
        VALUES (
            :male_id, :female_id, :operator_id, :operator_name, :shift_key,
            :man_name, :man_age, :man_city,
            :woman_name, :woman_age, :woman_city,
            :text, :updated_at, :actions_total, :actions_paid, :balance_earned
        )
        ON CONFLICT(male_id, female_id, operator_id, shift_key)
        DO UPDATE SET
            operator_name=excluded.operator_name,
            shift_key=excluded.shift_key,
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


def upsert_chat_spend_max(
    conn: sqlite3.Connection, payload: ChatSpendUpsertPayload
) -> Dict[str, Any]:
    male_id = payload.male_id.strip()
    female_id = payload.female_id.strip()
    if not male_id or not female_id:
        raise HTTPException(status_code=400, detail="male_id and female_id are required")
    if not TEN_DIGITS.match(male_id):
        raise HTTPException(status_code=400, detail="male_id must be 10 digits")
    incoming_max = max(0.0, float(payload.max_spend_all_credits or 0.0))
    updated_at = int(payload.updated_at or 0)
    if updated_at <= 0:
        updated_at = int(time.time() * 1000)
    chat_uid = (payload.chat_uid or "").strip() or None
    last_operator_id = (payload.operator_id or "").strip() or None
    last_operator_name = (payload.operator_name or "").strip() or None

    row = conn.execute(
        """
        SELECT max_spend_all_credits
        FROM chat_spend_max
        WHERE male_id = ? AND female_id = ?
        """,
        (male_id, female_id),
    ).fetchone()
    prev_max = float(row["max_spend_all_credits"] or 0.0) if row else 0.0
    changed = incoming_max > prev_max

    conn.execute(
        """
        INSERT INTO chat_spend_max (
            male_id,
            female_id,
            max_spend_all_credits,
            chat_uid,
            last_operator_id,
            last_operator_name,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(male_id, female_id)
        DO UPDATE SET
            max_spend_all_credits = CASE
                WHEN excluded.max_spend_all_credits > chat_spend_max.max_spend_all_credits
                THEN excluded.max_spend_all_credits
                ELSE chat_spend_max.max_spend_all_credits
            END,
            updated_at = CASE
                WHEN excluded.max_spend_all_credits > chat_spend_max.max_spend_all_credits
                THEN excluded.updated_at
                ELSE chat_spend_max.updated_at
            END,
            chat_uid = CASE
                WHEN excluded.max_spend_all_credits > chat_spend_max.max_spend_all_credits
                THEN excluded.chat_uid
                ELSE chat_spend_max.chat_uid
            END,
            last_operator_id = CASE
                WHEN excluded.max_spend_all_credits > chat_spend_max.max_spend_all_credits
                THEN excluded.last_operator_id
                ELSE chat_spend_max.last_operator_id
            END,
            last_operator_name = CASE
                WHEN excluded.max_spend_all_credits > chat_spend_max.max_spend_all_credits
                THEN excluded.last_operator_name
                ELSE chat_spend_max.last_operator_name
            END
        """,
        (
            male_id,
            female_id,
            incoming_max,
            chat_uid,
            last_operator_id,
            last_operator_name,
            updated_at,
        ),
    )

    current = conn.execute(
        """
        SELECT
            max_spend_all_credits,
            updated_at,
            chat_uid,
            last_operator_id,
            last_operator_name
        FROM chat_spend_max
        WHERE male_id = ? AND female_id = ?
        """,
        (male_id, female_id),
    ).fetchone()
    stored_max = (
        float(current["max_spend_all_credits"] or 0.0)
        if current
        else max(incoming_max, prev_max)
    )
    return {
        "male_id": male_id,
        "female_id": female_id,
        "stored_max_spend_all_credits": stored_max,
        "chat_uid": (current["chat_uid"] if current else chat_uid) or None,
        "last_operator_id": (current["last_operator_id"] if current else last_operator_id) or None,
        "last_operator_name": (
            (current["last_operator_name"] if current else last_operator_name) or None
        ),
        "updated_at": int(current["updated_at"] or updated_at) if current else updated_at,
        "updated": bool(changed),
    }


def upsert_report_snapshot(
    conn: sqlite3.Connection,
    payload: ReportShiftSnapshotPayload,
) -> bool:
    updated_at = int(payload.updated_at or 0)
    if updated_at <= 0:
        updated_at = int(time.time() * 1000)
    raw_shift_key = (payload.shift_key or "").strip()
    shift_key = raw_shift_key or compute_shift_key(updated_at)
    fields = {
        "male_id": payload.male_id,
        "female_id": payload.female_id.strip(),
        "operator_id": payload.operator_id.strip(),
        "operator_name": (payload.operator_name or "").strip() or None,
        "shift_key": shift_key,
        "text": payload.text or "",
        "updated_at": updated_at,
    }
    cur = conn.execute(
        """
        INSERT INTO reports (
            male_id, female_id, operator_id, operator_name, shift_key,
            text, updated_at
        )
        VALUES (
            :male_id, :female_id, :operator_id, :operator_name, :shift_key,
            :text, :updated_at
        )
        ON CONFLICT(male_id, female_id, operator_id, shift_key)
        DO UPDATE SET
            operator_name=excluded.operator_name,
            text=excluded.text,
            updated_at=excluded.updated_at
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


@app.post("/api/reports/shift/snapshot")
def save_report_shift_snapshot(
    payload: ReportShiftSnapshotPayload,
    _=Depends(require_latest_extension_version),
):
    conn = get_conn()
    try:
        changed = False
        with conn:
            changed = upsert_report_snapshot(conn, payload)
        return {"ok": True, "updated": 1 if changed else 0}
    finally:
        conn.close()


@app.get("/api/reports/shift")
def get_report_shift(
    male_id: str,
    female_id: str,
    day_key: Optional[str] = None,
    _=Depends(require_latest_extension_version),
):
    male_id = (male_id or "").strip()
    female_id = (female_id or "").strip()
    if not TEN_DIGITS.match(male_id):
        raise HTTPException(status_code=400, detail="male_id must be exactly 10 digits")
    if not female_id:
        raise HTTPException(status_code=400, detail="female_id is required")
    conn = get_conn()
    try:
        normalized_day = (day_key or "").strip()
        if normalized_day and normalized_day.lower() != "all":
            shift_key = normalize_state_day_key(day_key)
            cur = conn.execute(
                """
                SELECT male_id, female_id, operator_id, operator_name, shift_key, text, updated_at
                FROM reports
                WHERE male_id = ? AND female_id = ? AND shift_key = ?
                ORDER BY updated_at DESC, operator_id ASC
                """,
                (male_id, female_id, shift_key),
            )
        else:
            shift_key = None
            cur = conn.execute(
                """
                SELECT male_id, female_id, operator_id, operator_name, shift_key, text, updated_at
                FROM reports
                WHERE male_id = ? AND female_id = ?
                ORDER BY updated_at DESC, operator_id ASC
                """,
                (male_id, female_id),
            )
        rows = [dict(row) for row in cur.fetchall()]
        return {"ok": True, "shift_key": shift_key, "items": rows}
    finally:
        conn.close()


@app.get("/api/reports/shift/exists")
def has_report_shift(
    male_id: str,
    female_id: str,
    day_key: Optional[str] = None,
    _=Depends(require_latest_extension_version),
):
    male_id = (male_id or "").strip()
    female_id = (female_id or "").strip()
    if not TEN_DIGITS.match(male_id):
        raise HTTPException(status_code=400, detail="male_id must be exactly 10 digits")
    if not female_id:
        raise HTTPException(status_code=400, detail="female_id is required")
    conn = get_conn()
    try:
        normalized_day = (day_key or "").strip()
        if normalized_day and normalized_day.lower() != "all":
            shift_key = normalize_state_day_key(day_key)
            cur = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM reports
                WHERE male_id = ? AND female_id = ? AND shift_key = ?
                """,
                (male_id, female_id, shift_key),
            )
        else:
            shift_key = None
            cur = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM reports
                WHERE male_id = ? AND female_id = ?
                """,
                (male_id, female_id),
            )
        row = cur.fetchone()
        count = int(row["c"] if row and row["c"] is not None else 0)
        return {"ok": True, "shift_key": shift_key, "count": count, "exists": count > 0}
    finally:
        conn.close()


@app.post("/api/operator/state")
def save_operator_state(payload: OperatorStatePayload, _=Depends(auth)):
    operator_id = payload.operator_id.strip()
    if not operator_id:
        raise HTTPException(status_code=400, detail="operator_id is required")
    day_key = normalize_state_day_key(payload.day_key)
    incoming_sections = payload.sections or {}
    filtered: Dict[str, StateSectionPayload] = {}
    for name, section_payload in incoming_sections.items():
        key = (name or "").strip()
        if key in STATE_SECTIONS:
            filtered[key] = section_payload
    global_filtered: Dict[str, StateSectionPayload] = {}
    for name, section_payload in (payload.global_sections or {}).items():
        key = (name or "").strip()
        if key in GLOBAL_STATE_SECTIONS:
            global_filtered[key] = section_payload
    if not filtered and not global_filtered:
        return {
            "ok": True,
            "operator_id": operator_id,
            "day_key": day_key,
            "updated_sections": 0,
        }
    conn = get_conn()
    try:
        updated = 0
        with conn:
            for section_name, section_payload in filtered.items():
                changed = upsert_state_section(
                    conn,
                    operator_id,
                    day_key,
                    section_name,
                    int(section_payload.updated_at),
                    section_payload.data,
                )
                if changed:
                    apply_section_side_effects(
                        conn,
                        section_name,
                        section_payload.data,
                        day_key,
                    )
                    updated += 1
            for section_name, section_payload in global_filtered.items():
                changed = upsert_state_section(
                    conn,
                    operator_id,
                    day_key,
                    section_name,
                    int(section_payload.updated_at),
                    section_payload.data,
                )
                if changed:
                    updated += 1
        return {
            "ok": True,
            "operator_id": operator_id,
            "day_key": day_key,
            "updated_sections": updated,
        }
    finally:
        conn.close()


@app.get("/api/operator/state")
def get_operator_state(
    operator_id: str,
    day_key: Optional[str] = None,
    sections: Optional[str] = None,
    _=Depends(auth),
):
    operator_id = (operator_id or "").strip()
    if not operator_id:
        raise HTTPException(status_code=400, detail="operator_id is required")
    day_key_value = normalize_state_day_key(day_key)
    local_sections: Optional[List[str]] = None
    global_sections_filter: Optional[List[str]] = None
    if sections:
        for part in sections.split(","):
            normalized = part.strip()
            if not normalized:
                continue
            if normalized in STATE_SECTIONS:
                if local_sections is None:
                    local_sections = []
                local_sections.append(normalized)
            elif normalized in GLOBAL_STATE_SECTIONS:
                if global_sections_filter is None:
                    global_sections_filter = []
                global_sections_filter.append(normalized)
        if local_sections is not None and not local_sections:
            local_sections = None
        if global_sections_filter is not None and not global_sections_filter:
            global_sections_filter = None
    conn = get_conn()
    try:
        data = fetch_state_sections(conn, operator_id, day_key_value, local_sections)
        history_requested = local_sections is None or (
            isinstance(local_sections, list) and "history" in local_sections
        )
        if history_requested:
            if "history" in data:
                history_payload = data["history"].get("data")
                data["history"]["data"] = enrich_history_section_with_hourly(
                    conn,
                    history_payload,
                    day_key_value,
                    operator_id,
                )
            else:
                data["history"] = {
                    "updated_at": int(time.time() * 1000),
                    "data": enrich_history_section_with_hourly(
                        conn,
                        {},
                        day_key_value,
                        operator_id,
                    ),
                }
        if sections:
            global_filter = global_sections_filter or []
        else:
            global_filter = list(GLOBAL_STATE_SECTIONS)
        if global_filter:
            global_data = fetch_state_sections(
                conn,
                "__GLOBAL__",
                day_key_value,
                [s for s in global_filter if s != "operator_names"],
            )
            if "operator_names" in global_filter:
                names_data = fetch_state_sections(
                    conn,
                    "__GLOBAL__",
                    GLOBAL_OPERATOR_NAMES_DAY_KEY,
                    ["operator_names"],
                )
                global_data.update(names_data)
            data.update(global_data)
        return {
            "ok": True,
            "operator_id": operator_id,
            "day_key": day_key_value,
            "sections": data,
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
        # aggregate shift summary
        summary = {
            "female_id": female_id,
            "shift_key": key,
            "balance_earned": 0.0,
            "actions_total": 0,
            "actions_paid": 0,
            "chat_count": 0,
            "mail_count": 0,
            "operator_summary": {},
        }
        for row in rows:
            summary["balance_earned"] += float(row.get("balance_earned") or 0.0)
            summary["actions_total"] += int(row.get("actions_total") or 0)
            summary["actions_paid"] += int(row.get("actions_paid") or 0)
            summary["chat_count"] += int(row.get("chat_count") or 0)
            summary["mail_count"] += int(row.get("mail_count") or 0)
            op_id = row.get("operator_id") or ""
            op_name = row.get("operator_name") or ""
            op_entry = summary["operator_summary"].setdefault(op_id, {
                "operator_id": op_id,
                "operator_name": op_name,
                "actions_total": 0,
                "balance_earned": 0.0,
            })
            op_entry["actions_total"] += int(row.get("actions_total") or 0)
            op_entry["balance_earned"] += float(row.get("balance_earned") or 0.0)
        summary["operator_summary"] = [
            value for key, value in summary["operator_summary"].items()
        ]
        return {
            "ok": True,
            "items": rows,
            "shift_key": key,
            "shift_summary": summary,
        }
    finally:
        conn.close()


@app.get("/api/profiles/stats")
def get_profiles_stats(day_key: Optional[str] = None, _=Depends(auth)):
    """
    Сводные статистики по анкетам за смену:
    суммируем действия всех операторов по каждой анкете.
    """
    start_ms, end_ms = get_kyiv_day_range(day_key)
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT
                female_id,
                SUM(chat_count) AS chat_count,
                SUM(mail_count) AS mail_count,
                SUM(actions_total) AS actions_total
            FROM hourly_stats
            WHERE hour_start >= ? AND hour_start < ?
            GROUP BY female_id
            """,
            (start_ms, end_ms),
        )
        rows = [
            {
                "female_id": row["female_id"],
                "chat_count": int(row["chat_count"] or 0),
                "mail_count": int(row["mail_count"] or 0),
                "actions_total": int(row["actions_total"] or 0),
            }
            for row in cur.fetchall()
        ]
        return {"ok": True, "profiles": rows, "day_key": day_key or None}
    finally:
        conn.close()


@app.post("/api/profiles/shift/delta")
def sync_profile_shift_delta(
    payload: ProfileShiftDeltaPayload,
    _=Depends(require_latest_extension_version),
):
    if not payload.profiles:
        return {"ok": True, "updated": 0}
    day_key = normalize_operator_shift_day_key(payload.day_key)
    conn = get_conn()
    try:
        updated = 0
        with conn:
            for entry in payload.profiles:
                try:
                    if apply_profile_shift_delta(conn, entry, day_key):
                        updated += 1
                except Exception:
                    continue
        return {"ok": True, "day_key": day_key, "updated": updated}
    finally:
        conn.close()


@app.post("/api/profiles/shift/batch")
def get_profile_shift_batch(
    payload: ProfileShiftBatchPayload,
    _=Depends(require_latest_extension_version),
):
    day_key = normalize_operator_shift_day_key(payload.day_key)
    conn = get_conn()
    try:
        items = fetch_profile_shift_stats(conn, day_key, payload.profile_ids)
        return {"ok": True, "day_key": day_key, "profiles": items}
    finally:
        conn.close()


@app.post("/api/operators/shift/snapshot")
def save_operator_shift_snapshot(
    payload: OperatorShiftSnapshotPayload,
    _=Depends(require_latest_extension_version),
):
    conn = get_conn()
    try:
        changed = False
        with conn:
            changed = upsert_operator_shift_summary(conn, payload)
            op_name = (payload.operator_name or "").strip()
            if op_name:
                upsert_state_section(
                    conn,
                    payload.operator_id.strip(),
                    compute_operator_shift_day_key(int(payload.updated_at or 0)),
                    "operator_names",
                    int(payload.updated_at or 0),
                    [
                        {
                            "operator_id": payload.operator_id.strip(),
                            "operator_name": op_name,
                            "updated_at": int(payload.updated_at or 0),
                        }
                    ],
                )
        return {"ok": True, "updated": 1 if changed else 0}
    finally:
        conn.close()


@app.post("/api/audio/transcribe")
def transcribe_audio(
    payload: AudioTranscribePayload,
    _=Depends(require_latest_extension_version),
):
    try:
        audio_url = normalize_audio_transcribe_url(payload.audio_url)
        audio_bytes = download_audio_transcribe_bytes(audio_url)
        transcript = transcribe_audio_bytes(audio_url, audio_bytes)
        return {"ok": True, "audio_url": audio_url, "transcript": transcript}
    except AudioTranscribeError as exc:
        return {"ok": False, "error": exc.code}


@app.get("/api/audio/voices")
def get_audio_voices(
    _=Depends(require_latest_extension_version),
):
    return {"ok": True, "voices": get_elevenlabs_voice_list()}


@app.get("/api/audio/pair-selection")
def get_audio_pair_selection(
    male_id: str,
    female_id: str,
    _=Depends(require_latest_extension_version),
):
    normalized_male_id = str(male_id or "").strip()
    normalized_female_id = str(female_id or "").strip()
    if not TEN_DIGITS.fullmatch(normalized_male_id):
        raise HTTPException(status_code=400, detail="male_id is invalid")
    if not normalized_female_id:
        raise HTTPException(status_code=400, detail="female_id is required")
    conn = get_conn()
    try:
        selection = fetch_audio_pair_selection(conn, normalized_male_id, normalized_female_id)
        return {"ok": True, "selection": selection}
    finally:
        conn.close()


@app.get("/api/audio/generation-quota")
def get_audio_generation_quota(
    operator_id: str,
    _=Depends(require_latest_extension_version),
):
    normalized_operator_id = str(operator_id or "").strip()
    if not normalized_operator_id:
        raise HTTPException(status_code=400, detail="operator_id is required")
    conn = get_conn()
    try:
        return {"ok": True, **build_audio_generation_quota_payload(conn, normalized_operator_id)}
    finally:
        conn.close()


@app.post("/api/audio/pair-selection")
def save_audio_pair_selection(
    payload: AudioPairSelectionPayload,
    _=Depends(require_latest_extension_version),
):
    conn = get_conn()
    try:
        with conn:
            updated = upsert_audio_pair_selection(conn, payload)
        selection = fetch_audio_pair_selection(conn, payload.male_id, payload.female_id)
        return {"ok": True, "updated": 1 if updated else 0, "selection": selection}
    finally:
        conn.close()


@app.post("/api/audio/generate")
def generate_audio(
    payload: AudioGeneratePayload,
    _=Depends(require_latest_extension_version),
):
    conn = get_conn()
    try:
        quota = build_audio_generation_quota_payload(conn, payload.operator_id)
        if quota["blocked"]:
            return JSONResponse(
                status_code=429,
                content={
                    "ok": False,
                    "error": "generation_limit_reached",
                    "limit": quota["limit"],
                    "used": quota["used"],
                    "remaining": quota["remaining"],
                },
            )
        voice_key, mood = resolve_elevenlabs_generate_selection(payload)
        audio_bytes, filename = generate_elevenlabs_audio_bytes(
            payload.text,
            voice_key,
            mood,
        )
        now_ms = int(time.time() * 1000)
        with conn:
            quota = increment_audio_generation_usage(
                conn,
                payload.operator_id,
                payload.operator_name,
                payload.agency_id,
                now_ms,
            )
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Audio-Filename": filename,
                "X-Audio-Generation-Limit": str(quota["limit"]),
                "X-Audio-Generation-Used": str(quota["used"]),
                "X-Audio-Generation-Remaining": str(quota["remaining"]),
            },
        )
    except AudioGenerateError as exc:
        return JSONResponse(
            status_code=400
            if exc.code
            in {
                "invalid_preset",
                "invalid_voice_key",
                "invalid_mood",
                "voice_not_configured",
                "elevenlabs_voice_unavailable",
                "elevenlabs_invalid_voice_settings",
            }
            else 502,
            content={"ok": False, "error": exc.code},
        )
    finally:
        conn.close()


@app.post("/api/auth/check")
def check_extension_password(
    payload: ExtensionAuthPayload,
    x_extension_version: str | None = Header(default=None),
    _=Depends(require_latest_extension_version),
):
    raw_password = (payload.password or "").strip()
    if not raw_password:
        return {"ok": False}
    install_id = (payload.install_id or "").strip()
    operator_id = (payload.operator_id or "").strip()
    agency_id = (payload.agency_id or "").strip()
    extension_version = (x_extension_version or "").strip()
    count_success = bool(payload.count_success)
    is_admin_context = bool(payload.is_admin_context)
    conn = get_conn()
    try:
        active_count = active_extension_passwords_count(conn)
        if active_count > 0:
            row = fetch_active_extension_password(conn, raw_password)
            if not row:
                return {"ok": False}
            response_payload: Dict[str, Any] = {"ok": True}
            if install_id or operator_id:
                now_ms = int(time.time() * 1000)
                with conn:
                    if install_id:
                        upsert_extension_password_usage(
                            conn,
                            int(row["id"]),
                            install_id,
                            now_ms,
                            extension_version=extension_version,
                            count_success=count_success,
                        )
                    if install_id and operator_id:
                        upsert_extension_password_operator_usage(
                            conn,
                            int(row["id"]),
                            install_id,
                            operator_id,
                            agency_id,
                            now_ms,
                            count_success=count_success,
                        )
                    team_name = str(row["team_name"] or "").strip()
                    if operator_id and team_name:
                        upsert_operator_team_state(
                            conn,
                            operator_id,
                            team_name,
                            int(row["id"]),
                            now_ms,
                        )
                    install_registry_row = None
                    binding_row = None
                    if install_id:
                        install_registry_row = upsert_extension_install_registry(
                            conn,
                            install_id,
                            now_ms,
                            extension_version=extension_version,
                            operator_id=operator_id or None,
                            agency_id=agency_id or None,
                        )
                        if is_admin_context:
                            install_registry_row = mark_install_as_admin(
                                conn,
                                install_id,
                                now_ms,
                                "client_admin_context",
                                operator_id=operator_id or None,
                                agency_id=agency_id or None,
                                extension_version=extension_version,
                            )
                            remove_bindings_for_install(conn, install_id, now_ms)
                    if install_id and operator_id:
                        upsert_install_operator_history(
                            conn,
                            install_id,
                            operator_id,
                            agency_id or None,
                            now_ms,
                            is_admin_context=is_admin_context,
                        )
                        distinct_operator_count = count_distinct_install_operator_ids(
                            conn, install_id
                        )
                        admin_reason = ""
                        if is_admin_context:
                            admin_reason = "client_admin_context"
                        elif distinct_operator_count > 1:
                            admin_reason = "multiple_operator_ids"
                        if admin_reason and install_registry_row is not None:
                            install_registry_row = mark_install_as_admin(
                                conn,
                                install_id,
                                now_ms,
                                admin_reason,
                                operator_id=operator_id,
                                agency_id=agency_id or None,
                                extension_version=extension_version,
                            )
                            remove_bindings_for_install(conn, install_id, now_ms)
                        if install_registry_row is not None and int(
                            install_registry_row["is_admin_install"] or 0
                        ):
                            response_payload.update(
                                {
                                    "mode": "admin_marked",
                                    "active_install_id": "",
                                    "replaced_previous": False,
                                    "reason": str(
                                        install_registry_row["admin_reason"] or ""
                                    ).strip(),
                                }
                            )
                        else:
                            binding_row = fetch_operator_install_binding(conn, operator_id)
                            current_first_seen = int(
                                install_registry_row["first_seen_at"] or 0
                            ) if install_registry_row is not None else 0
                            if not binding_row:
                                binding_row, replaced_previous = upsert_operator_install_binding(
                                    conn,
                                    operator_id,
                                    install_id,
                                    now_ms,
                                    extension_version=extension_version,
                                    agency_id=agency_id or None,
                                    reason="first_bind",
                                )
                                response_payload.update(
                                    {
                                        "mode": "operator_bound",
                                        "active_install_id": str(
                                            binding_row["active_install_id"] or ""
                                        ).strip(),
                                        "replaced_previous": replaced_previous,
                                        "reason": "",
                                    }
                                )
                            else:
                                active_install_id = str(
                                    binding_row["active_install_id"] or ""
                                ).strip()
                                binding_updated_at = int(binding_row["updated_at"] or 0)
                                if active_install_id == install_id:
                                    binding_row, replaced_previous = (
                                        upsert_operator_install_binding(
                                            conn,
                                            operator_id,
                                            install_id,
                                            now_ms,
                                            extension_version=extension_version,
                                            agency_id=agency_id or None,
                                            reason="same_install",
                                        )
                                    )
                                    response_payload.update(
                                        {
                                            "mode": "operator_bound",
                                            "active_install_id": str(
                                                binding_row["active_install_id"] or ""
                                            ).strip(),
                                            "replaced_previous": replaced_previous,
                                            "reason": "",
                                        }
                                    )
                                elif current_first_seen > binding_updated_at:
                                    binding_row, replaced_previous = (
                                        upsert_operator_install_binding(
                                            conn,
                                            operator_id,
                                            install_id,
                                            now_ms,
                                            extension_version=extension_version,
                                            agency_id=agency_id or None,
                                            reason="new_install",
                                        )
                                    )
                                    response_payload.update(
                                        {
                                            "mode": "operator_bound",
                                            "active_install_id": str(
                                                binding_row["active_install_id"] or ""
                                            ).strip(),
                                            "replaced_previous": replaced_previous,
                                            "reason": "",
                                        }
                                    )
                                else:
                                    response_payload.update(
                                        {
                                            "ok": False,
                                            "mode": "operator_bound",
                                            "active_install_id": active_install_id,
                                            "replaced_previous": False,
                                            "reason": "install_replaced",
                                        }
                                    )
                    elif install_registry_row is not None and int(
                        install_registry_row["is_admin_install"] or 0
                    ):
                        response_payload.update(
                            {
                                "mode": "admin_marked",
                                "active_install_id": "",
                                "replaced_previous": False,
                                "reason": str(
                                    install_registry_row["admin_reason"] or ""
                                ).strip(),
                            }
                        )
            return response_payload
    finally:
        conn.close()

    ok = bool(EXTENSION_ACCESS_PASSWORD) and raw_password == EXTENSION_ACCESS_PASSWORD
    return {"ok": ok}


@app.get("/api/operators/shift")
def get_operator_shift_snapshot(
    operator_id: str,
    day_key: Optional[str] = None,
    _=Depends(require_latest_extension_version),
):
    operator_id = (operator_id or "").strip()
    if not operator_id:
        raise HTTPException(status_code=400, detail="operator_id is required")
    target_day = normalize_operator_shift_day_key(day_key)
    conn = get_conn()
    try:
        data = get_operator_shift_summary(conn, target_day, operator_id)
        return {"ok": True, "day_key": target_day, "operator_id": operator_id, "summary": data}
    finally:
        conn.close()


@app.post("/api/chat/spend/upsert")
def save_chat_spend_max(payload: ChatSpendUpsertPayload, _=Depends(auth)):
    conn = get_conn()
    try:
        with conn:
            data = upsert_chat_spend_max(conn, payload)
        return {
            "ok": True,
            "male_id": data["male_id"],
            "female_id": data["female_id"],
            "stored_max_spend_all_credits": data["stored_max_spend_all_credits"],
            "updated_at": data["updated_at"],
            "updated": data["updated"],
        }
    finally:
        conn.close()


@app.get("/api/chat/spend/max")
def get_chat_spend_max(male_id: str, female_id: str, _=Depends(auth)):
    normalized_male = (male_id or "").strip()
    normalized_female = (female_id or "").strip()
    if not normalized_male or not normalized_female:
        raise HTTPException(status_code=400, detail="male_id and female_id are required")
    if not TEN_DIGITS.match(normalized_male):
        raise HTTPException(status_code=400, detail="male_id must be 10 digits")
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT
                max_spend_all_credits,
                chat_uid,
                last_operator_id,
                last_operator_name,
                updated_at
            FROM chat_spend_max
            WHERE male_id = ? AND female_id = ?
            """,
            (normalized_male, normalized_female),
        ).fetchone()
        if not row:
            return {
                "ok": True,
                "exists": False,
                "male_id": normalized_male,
                "female_id": normalized_female,
                "max_spend_all_credits": 0,
                "chat_uid": None,
                "last_operator_id": None,
                "last_operator_name": None,
                "updated_at": 0,
            }
        return {
            "ok": True,
            "exists": True,
            "male_id": normalized_male,
            "female_id": normalized_female,
            "max_spend_all_credits": float(row["max_spend_all_credits"] or 0.0),
            "chat_uid": (row["chat_uid"] or None),
            "last_operator_id": (row["last_operator_id"] or None),
            "last_operator_name": (row["last_operator_name"] or None),
            "updated_at": int(row["updated_at"] or 0),
        }
    finally:
        conn.close()


@app.get("/api/chat/spend/total-by-male")
def get_chat_spend_total_by_male(male_id: str, _=Depends(auth)):
    normalized_male = (male_id or "").strip()
    if not TEN_DIGITS.match(normalized_male):
        raise HTTPException(status_code=400, detail="male_id must be 10 digits")
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(max_spend_all_credits), 0) AS total_spend,
                COUNT(*) AS pairs_count,
                MAX(updated_at) AS updated_at
            FROM chat_spend_max
            WHERE male_id = ?
            """,
            (normalized_male,),
        ).fetchone()
        total_spend = float((row["total_spend"] if row else 0) or 0.0)
        pairs_count = int((row["pairs_count"] if row else 0) or 0)
        updated_at = int((row["updated_at"] if row else 0) or 0)
        return {
            "ok": True,
            "male_id": normalized_male,
            "total_spend_all_credits": total_spend,
            "pairs_count": pairs_count,
            "updated_at": updated_at,
        }
    finally:
        conn.close()


def merge_global_top_entries(
    existing_entries: List[Dict[str, Any]],
    incoming_entries: List[Dict[str, Any]],
    default_day_key: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    def normalize_operator(entry: Dict[str, Any]) -> str:
        raw = entry.get("operator_id") or entry.get("operatorId") or ""
        return str(raw).strip()

    def normalize_day(entry: Dict[str, Any]) -> str:
        raw = entry.get("day_key") or entry.get("dayKey") or ""
        return normalize_state_day_key(raw) or default_day_key

    def parse_number(value: Any) -> Optional[float]:
        try:
            num = float(value)
            if math.isfinite(num):
                return num
        except (TypeError, ValueError):
            return None
        return None

    def entry_updated(entry: Optional[Dict[str, Any]]) -> int:
        if not entry:
            return 0
        return int(entry.get("updated_at") or entry.get("updatedAt") or 0)

    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in existing_entries:
        operator_id = normalize_operator(entry)
        day_key = normalize_day(entry)
        if not operator_id or not day_key:
            continue
        entry_copy = dict(entry)
        entry_copy["day_key"] = day_key
        merged[(operator_id, day_key)] = entry_copy
    changed = False
    for entry in incoming_entries:
        operator_id = normalize_operator(entry)
        day_key = normalize_day(entry)
        if not operator_id or not day_key:
            continue
        key = (operator_id, day_key)
        current = merged.get(key)
        new_shift = parse_number(entry.get("shift_balance") or entry.get("shiftBalance"))
        new_hour = parse_number(entry.get("hour_balance") or entry.get("hourBalance"))
        new_name = (entry.get("operator_name") or entry.get("operatorName") or "").strip()
        new_ts = entry_updated(entry) or int(time.time() * 1000)
        if not current:
            merged[key] = {
                "operator_id": operator_id,
                "operator_name": new_name,
                "shift_balance": new_shift if new_shift is not None else 0,
                "hour_balance": new_hour if new_hour is not None else 0,
                "updated_at": new_ts,
                "day_key": day_key,
            }
            changed = True
            continue
        updated = False
        if new_shift is not None:
            current_shift = parse_number(current.get("shift_balance"))
            if current_shift is None or new_shift > current_shift:
                current["shift_balance"] = new_shift
                updated = True
        if new_hour is not None:
            current_hour = parse_number(current.get("hour_balance"))
            if current_hour is None or new_hour > current_hour:
                current["hour_balance"] = new_hour
                updated = True
        if new_name and new_name != current.get("operator_name"):
            current["operator_name"] = new_name
            updated = True
        if updated:
            current["updated_at"] = max(current.get("updated_at") or 0, new_ts)
            changed = True
    return list(merged.values()), changed


def merge_global_operator_names(
    existing_entries: List[Dict[str, Any]],
    incoming_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    def normalize_operator(entry: Dict[str, Any]) -> str:
        raw = entry.get("operator_id") or entry.get("operatorId") or ""
        return str(raw).strip()

    def entry_updated(entry: Optional[Dict[str, Any]]) -> int:
        if not entry:
            return 0
        return int(entry.get("updated_at") or entry.get("updatedAt") or 0)

    merged: Dict[str, Dict[str, Any]] = {}
    for entry in existing_entries:
        operator_id = normalize_operator(entry)
        if operator_id:
            merged[operator_id] = dict(entry)
    changed = False
    for entry in incoming_entries:
        operator_id = normalize_operator(entry)
        if not operator_id:
            continue
        new_ts = entry_updated(entry) or int(time.time() * 1000)
        current = merged.get(operator_id)
        if not current or new_ts >= entry_updated(current):
            entry_copy = dict(entry)
            entry_copy["operator_id"] = operator_id
            entry_copy["updated_at"] = new_ts
            merged[operator_id] = entry_copy
            changed = True
    return list(merged.values()), changed
