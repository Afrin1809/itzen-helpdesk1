# app.py -- ITZen-AI backend with SQLite ticket persistence (SQLAlchemy)
# Extended: safe static serving (Option A) so POST /api/* is not intercepted by StaticFiles

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import pytz
import random
import re
import logging
import shutil
import uuid
import os

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, String, Text, Integer
from sqlalchemy.orm import declarative_base, sessionmaker

# ----- Logging -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("itzen-backend")

# ----- Paths -----
BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "kbv.json"
# frontend is a sibling of backend (project root contains `frontend/` and `backend/`)
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ----- App -----
app = FastAPI(title="ITZen-AI Helpdesk API", version="1.1")

# For local dev allow all origins; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ----- Utility functions -----
def now_ist_formatted() -> str:
    utc = pytz.utc
    ist = pytz.timezone("Asia/Kolkata")
    now_utc = datetime.now(utc)
    now_ist = now_utc.astimezone(ist)
    return now_ist.strftime("%d-%m-%Y %I:%M %p")

def now_ist_for_id() -> str:
    utc = pytz.utc
    ist = pytz.timezone("Asia/Kolkata")
    now_utc = datetime.now(utc)
    now_ist = now_utc.astimezone(ist)
    return now_ist.strftime("%H%M%S")

def make_id_from_title(title: str) -> str:
    id = re.sub(r'[^a-z0-9_]+', '_', title.lower()).strip('_')
    if not id:
        id = f"kb_{int(datetime.utcnow().timestamp())}"
    return id

def save_kb_to_disk(kb: Dict[str, Any]) -> bool:
    try:
        KB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with KB_PATH.open("w", encoding="utf-8") as f:
            json.dump(kb, f, indent=2, ensure_ascii=False)
        logger.info("Saved KB to %s (%d entries)", KB_PATH, len(kb))
        return True
    except Exception as e:
        logger.exception("Failed to save KB to disk: %s", e)
        return False

# ----- Load KB -----
def load_kb() -> Dict[str, Any]:
    if not KB_PATH.exists():
        logger.warning("KB file not found at %s", KB_PATH)
        return {}
    try:
        with KB_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("KB file parsed but top-level is not a dict.")
                return {}
            logger.info("Loaded KB with %d top-level entries.", len(data))
            return data
    except Exception as e:
        logger.exception("Failed to load KB from %s: %s", KB_PATH, e)
        return {}

KB: Dict[str, Any] = load_kb()

def build_items(kb: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for key, obj in kb.items():
        title = obj.get("title", key) if isinstance(obj, dict) else key
        issues = obj.get("issues", {}) if isinstance(obj, dict) else {}
        items.append({"id": key, "title": title, "issues": issues})
    return items

ITEMS = build_items(KB)

# ----- SQLite (SQLAlchemy) setup -----
# ----- PostgreSQL (SQLAlchemy) setup -----
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    # Render / Production (PostgreSQL)
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        echo=False,
        future=True
    )
else:
    # Local development (SQLite)
    DB_PATH = BASE_DIR / "tickets.db"
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        echo=False,
        future=True
    )

Base = declarative_base()
class Ticket(Base):
    __tablename__ = "tickets"
    ticket_id = Column(String, primary_key=True, index=True)
    zid = Column(String, nullable=False)          # ✅ NEW
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String, default="open")
    created_at = Column(String, nullable=False)


class SearchHistory(Base):
    __tablename__ = "search_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    term = Column(String, nullable=False, index=True)
    created_at = Column(String, nullable=False)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def ticket_to_dict(t: Ticket) -> Dict[str, Any]:
    return {
        "ticket_id": t.ticket_id,
        "title": t.title,
        "description": t.description,
        "status": t.status,
        "created_at": t.created_at,
    }

def search_row_to_dict(r: SearchHistory):
    return {"id": r.id, "term": r.term, "created_at": r.created_at}

def chat_row_to_dict(r: ChatMessage):
    return {"id": r.id, "role": r.role, "content": r.content, "created_at": r.created_at}

# ----- API Endpoints (read-only / tickets / chat) -----
@app.get("/api/kb")
async def get_kb():
    return KB

@app.get("/api/items")
async def get_items():
    return ITEMS

# --- Admin login endpoint (server-side auth) ---
@app.post("/api/admin/login")
async def admin_login(payload: Dict[str, str] = Body(...)):
    """
    Simple admin login endpoint for the frontend upload page.
    Expects JSON: { "username": "...", "password": "..." }
    Uses environment variables ADMIN_USER and ADMIN_PASS if set; otherwise
    defaults to Admin / Admin123 (change for production).
    Returns 200 on success, 401 on failure.
    """
    try:
        u = (payload.get("username") or "").strip()
        p = (payload.get("password") or "").strip()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")

    ADMIN_USER = os.environ.get("ADMIN_USER", "Admin")
    ADMIN_PASS = os.environ.get("ADMIN_PASS", "Admin123")

    # constant time compare could be used; equality is fine for this simple demo
    if u == ADMIN_USER and p == ADMIN_PASS:
        return {"ok": True}
    raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/api/search")
async def search(q: str = ""):
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing query parameter 'q'")
    for it in ITEMS:
        if it["title"].strip().lower() == q.lower():
            return {"exact": it, "related": []}
    for it in ITEMS:
        if q.lower() in it["title"].lower():
            return {"exact": it, "related": []}
    def tokenize(text: str):
        return [t for t in re.split(r"\W+", text.lower()) if t]
    def score_candidate(it: Dict[str, Any], tokens: List[str]) -> float:
        title = (it.get("title") or "").lower()
        issues_text = " ".join(list(it.get("issues", {}).keys()) + list(it.get("issues", {}).values())).lower()
        combined = f"{title} {issues_text}"
        score = 0.0
        for tok in tokens:
            if tok in title:
                score += 2.0
            if tok in issues_text:
                score += 1.0
            if len(tok) > 4 and tok in combined:
                score += 0.2
        return score
    tokens = tokenize(q)
    scored = [{"it": it, "score": score_candidate(it, tokens)} for it in ITEMS]
    scored.sort(key=lambda x: x["score"], reverse=True)
    related = [s["it"] for s in scored if s["score"] > 0][:12]
    best = related[0] if related else None
    return {"exact": best, "related": related}

# tickets endpoints
@app.post("/api/ticket")
async def create_ticket(payload: Dict[str, Any]):
    zid = (payload.get("zid") or "").strip()
    title = (payload.get("title") or payload.get("issue") or "Support request").strip()
    description = (payload.get("description") or "").strip()

    if not zid or not title:
        raise HTTPException(status_code=400, detail="ZID and Title required")

    created_at = now_ist_formatted()
    tid = f"HD-{random.randint(100,999)}-{now_ist_for_id()}"

    db = SessionLocal()
    try:
        t = Ticket(
            ticket_id=tid,
            zid=zid,
            title=title,
            description=description,
            status="open",
            created_at=created_at
        )
        db.add(t)
        db.commit()
        db.refresh(t)
        return JSONResponse(status_code=201, content=ticket_to_dict(t))
    except Exception as e:
        db.rollback()
        logger.exception("Failed to create ticket: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create ticket")
    finally:
        db.close()

@app.get("/api/tickets")
async def list_tickets():
    db = SessionLocal()
    try:
        rows = db.query(Ticket).order_by().all()
        return [ticket_to_dict(r) for r in rows]
    finally:
        db.close()

@app.get("/api/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    db = SessionLocal()
    try:
        t = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
        if not t:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return ticket_to_dict(t)
    finally:
        db.close()

# search history / chat endpoints
@app.get("/api/search-history")
async def get_search_history(limit: int = 200):
    db = SessionLocal()
    try:
        rows = db.query(SearchHistory).order_by(SearchHistory.id.desc()).limit(limit).all()
        return [search_row_to_dict(r) for r in rows]
    finally:
        db.close()

@app.post("/api/search-history")
async def post_search_history(payload: Dict[str, Any]):
    term = (payload.get("term") or "").strip()
    if not term:
        raise HTTPException(status_code=400, detail="Missing 'term'")
    created_at = now_ist_formatted()
    db = SessionLocal()
    try:
        r = SearchHistory(term=term, created_at=created_at)
        db.add(r)
        db.commit()
        db.refresh(r)
        return JSONResponse(status_code=201, content=search_row_to_dict(r))
    except Exception as e:
        db.rollback()
        logger.exception("Failed to save search history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save search history")
    finally:
        db.close()

@app.delete("/api/search-history")
async def clear_search_history():
    db = SessionLocal()
    try:
        db.query(SearchHistory).delete()
        db.commit()
        return {"cleared": True}
    finally:
        db.close()

@app.get("/api/chat-messages")
async def get_chat_messages(limit: int = 500):
    db = SessionLocal()
    try:
        rows = db.query(ChatMessage).order_by(ChatMessage.id.asc()).limit(limit).all()
        return [chat_row_to_dict(r) for r in rows]
    finally:
        db.close()

@app.post("/api/chat-message")
async def post_chat_message(payload: Dict[str, Any]):
    role = (payload.get("role") or "").strip()
    content = (payload.get("content") or "").strip()
    if not role or not content:
        raise HTTPException(status_code=400, detail="Missing 'role' or 'content'")
    created_at = now_ist_formatted()
    db = SessionLocal()
    try:
        m = ChatMessage(role=role, content=content, created_at=created_at)
        db.add(m)
        db.commit()
        db.refresh(m)
        return JSONResponse(status_code=201, content=chat_row_to_dict(m))
    except Exception as e:
        db.rollback()
        logger.exception("Failed to persist chat message: %s", e)
        raise HTTPException(status_code=500, detail="Failed to persist chat message")
    finally:
        db.close()

@app.delete("/api/chat-messages")
async def clear_chat_messages():
    db = SessionLocal()
    try:
        db.query(ChatMessage).delete()
        db.commit()
        return {"cleared": True}
    finally:
        db.close()

# ----- KB write / upload endpoint -----
@app.post("/api/kb/add")
async def api_kb_add(payload: Dict[str, Any]):
    global KB, ITEMS
    title = (payload.get("title") or "").strip()
    issues = payload.get("issues") or {}
    entry_id = (payload.get("id") or "").strip()

    if not title:
        raise HTTPException(status_code=400, detail="Missing 'title'")

    if not entry_id:
        entry_id = make_id_from_title(title)

    if entry_id in KB:
        i = 1
        base = entry_id
        while entry_id in KB:
            entry_id = f"{base}_{i}"
            i += 1

    if not isinstance(issues, dict):
        raise HTTPException(status_code=400, detail="'issues' must be an object/dict")

    KB[entry_id] = {"title": title, "issues": issues}
    ok = save_kb_to_disk(KB)
    ITEMS = build_items(KB)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save KB to disk; in-memory updated.")
    return {"updated": True, "id": entry_id, "title": title, "entries": len(KB)}

@app.post("/api/kb/save")
async def api_kb_save():
    ok = save_kb_to_disk(KB)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save KB to disk")
    return {"saved": True, "entries": len(KB)}

@app.get("/api/kb/download")
async def api_kb_download():
    if not KB_PATH.exists():
        ok = save_kb_to_disk(KB)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to create KB file for download")
    return FileResponse(path=str(KB_PATH), filename="kbv.json", media_type="application/json")

# Upload endpoint: accepts question + answer (text/file/link)
@app.post("/api/kb/upload")
async def api_kb_upload(
    question: str = Form(...),
    answer_type: str = Form(...),  # 'text' | 'file' | 'link'
    answer_text: Optional[str] = Form(None),
    answer_link: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    global KB, ITEMS

    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing question")

    typ = (answer_type or "").strip().lower()
    if typ not in ("text", "file", "link"):
        raise HTTPException(status_code=400, detail="answer_type must be 'text', 'file', or 'link'")

    issues: Dict[str, str] = {}

    if typ == "text":
        if not (answer_text and answer_text.strip()):
            raise HTTPException(status_code=400, detail="Missing answer_text for text answer type")
        issues["Solution"] = answer_text.strip()
    elif typ == "link":
        if not (answer_link and answer_link.strip()):
            raise HTTPException(status_code=400, detail="Missing answer_link for link answer type")
        issues["Solution"] = answer_link.strip()
    else:  # file
        if not file:
            raise HTTPException(status_code=400, detail="Missing uploaded file for file answer type")
        ext = Path(file.filename).suffix or ""
        uid = uuid.uuid4().hex
        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', Path(file.filename).stem)
        save_name = f"{safe_name}_{uid}{ext}"
        save_path = UPLOADS_DIR / save_name
        try:
            with save_path.open("wb") as out_f:
                shutil.copyfileobj(file.file, out_f)
            web_path = f"/uploads/{save_name}"
            issues["Solution"] = web_path
            logger.info("Saved uploaded file %s -> %s", file.filename, save_path)
        except Exception as e:
            logger.exception("Failed to save uploaded file: %s", e)
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        finally:
            try:
                file.file.close()
            except Exception:
                pass

    entry_id = make_id_from_title(q)
    if entry_id in KB:
        i = 1
        base = entry_id
        while entry_id in KB:
            entry_id = f"{base}_{i}"
            i += 1

    KB[entry_id] = {"title": q, "issues": issues}
    ok = save_kb_to_disk(KB)
    ITEMS = build_items(KB)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save KB to disk; in-memory updated.")
    return {"uploaded": True, "id": entry_id, "title": q, "issues": issues, "entries": len(KB)}

# Serve uploads publicly
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# ----- Static frontend serving (OPTION A) -----
# Serve static files under /static (optional), and serve index.html at root
if FRONTEND_DIR.exists() and FRONTEND_DIR.is_dir():
    logger.info("Mounting frontend static files from %s under /static", FRONTEND_DIR)
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file), media_type="text/html")
        return JSONResponse({"error": "index.html not found"}, status_code=404)

    # Catch-all GET: serve file if exists in frontend dir, else serve index (SPA-friendly)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        # Don't interfere with API routes (they're defined earlier)
        potential = FRONTEND_DIR / full_path
        if potential.exists() and potential.is_file():
            return FileResponse(str(potential))
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Not Found")
else:
    logger.warning("Frontend folder not found at %s. Static files will not be served.", FRONTEND_DIR)

# ----- Startup / Shutdown events -----
@app.on_event("startup")
async def on_startup():
    logger.info("ITZen backend starting up. KB entries: %d", len(KB))
    try:
        with engine.connect() as conn:
            pass
    except Exception as e:
        logger.exception("DB connection test failed: %s", e)

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("ITZen backend shutting down." )

# ----- If run as script -----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
