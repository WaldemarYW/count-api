import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

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
STATE_SECTIONS = {"reports", "hourly_stats", "chat_links", "history"}
GLOBAL_STATE_SECTIONS = {"top", "operator_names"}


def get_conn() -> sqlite3.Connection:
    """Возвращает соединение с основной БД (reports + hourly_stats)."""
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


def normalize_state_day_key(raw: Optional[str]) -> str:
    value = (raw or "").strip()
    if value:
        return value
    return compute_shift_key(int(time.time() * 1000))


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
        if sections:
            global_filter = global_sections_filter or []
        else:
            global_filter = list(GLOBAL_STATE_SECTIONS)
        if global_filter:
            global_data = fetch_state_sections(
                conn,
                "__GLOBAL__",
                day_key_value,
                global_filter,
            )
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
