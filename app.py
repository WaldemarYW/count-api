import os
import re
import sqlite3
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
DB_PATH = os.getenv("DB_PATH", "db.sqlite3")
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "*")

# Разрешённые источники CORS
if ALLOWED_ORIGINS_RAW.strip() == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in ALLOWED_ORIGINS_RAW.split(",") if o.strip()]

app = FastAPI(title="Count API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

TEN_DIGITS = re.compile(r"^\d{10}$")

def auth(authorization: str | None = Header(default=None)):
    if not API_KEY:
        return True  # ключ отключен
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True

def get_count_from_db(male_id: str) -> int:
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="DB file not found")
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) AS c
            FROM messages m
            JOIN message_male_ids mm ON mm.message_id_ref = m.id
            WHERE mm.male_id = ?
            """,
            (male_id,)
        )
        row = cur.fetchone()
        conn.close()
        return int(row["c"] if row and row["c"] is not None else 0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/count")
def count(male_id: str, _=Depends(auth)):
    # 1) Валидация
    if not TEN_DIGITS.match(male_id):
        raise HTTPException(status_code=400, detail="male_id must be exactly 10 digits")

    # 2) Достаём из БД
    n = get_count_from_db(male_id)

    # 3) Отдаём JSON
    return {"ok": True, "male_id": male_id, "count": n}

