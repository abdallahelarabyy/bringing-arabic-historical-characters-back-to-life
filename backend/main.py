"""FastAPI server: the question-answering API, text-to-speech, and the website.

Run from the project folder:
    python -m uvicorn backend.main:app --port 8000
(or double-click run.bat), then open http://localhost:8000
"""
import logging
import re
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.characters import CHARACTERS, Character, resolve_character
from backend.config import CONVERSATIONS_DIR, FRONTEND_DIR, INDEX_DIR, settings
from backend.schemas import SESSION_ID_PATTERN, AskRequest, SessionRequest, TTSRequest
from backend.services.llm import create_client
from backend.services.rag_engine import ArabicWikiRAG
from backend.services.tts import TTSError, TTSService, vocalize_words

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

rag: Optional[ArabicWikiRAG] = None
rag_ready = threading.Event()
startup_error: Optional[str] = None

llm = create_client(
    settings.llm_provider,
    groq_key=settings.groq_api_key, groq_models=settings.groq_models,
    gemini_key=settings.gemini_api_key, gemini_models=settings.gemini_models,
    openai_base_url=settings.openai_base_url, openai_key=settings.openai_api_key,
    openai_models=settings.openai_models,
)

tts = TTSService(
    engine=settings.tts_engine,
    elevenlabs_key=settings.elevenlabs_api_key,
    elevenlabs_voice=settings.elevenlabs_voice_id,
    elevenlabs_model=settings.elevenlabs_model,
    edge_rate=settings.edge_tts_rate,
)


def _load_rag() -> None:
    """Load the embedding model and indexes in the background so the website opens immediately."""
    global rag, startup_error
    try:
        engine = ArabicWikiRAG(
            index_dir=INDEX_DIR,
            conversations_dir=CONVERSATIONS_DIR,
            llm=llm,
            diacritics=settings.llm_diacritics,
            top_k=settings.rag_top_k,
        )
        engine.warmup()
        rag = engine
        log.info("RAG ready (LLM: %s)",
                 f"{llm.provider} / {', '.join(llm.models)}" if llm else "not configured - offline answers")
    except Exception as e:
        startup_error = str(e)
        log.exception("Startup failed")
    finally:
        rag_ready.set()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_load_rag, daemon=True).start()
    yield


app = FastAPI(title="Arabic Historical Characters API", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- helpers
def _character(name: str) -> Character:
    character = resolve_character(name)
    if not character:
        raise HTTPException(404, f"Unknown character: {name}. Available: {', '.join(CHARACTERS)}")
    return character


def _rag() -> ArabicWikiRAG:
    if not rag_ready.wait(timeout=300):
        raise HTTPException(503, "The server is still loading the AI models, try again in a minute.")
    if rag is None:
        raise HTTPException(503, f"The AI engine failed to start: {startup_error}")
    return rag


# ---------------------------------------------------------------- API
@app.get("/health", summary="Server status and configuration")
def health():
    if not rag_ready.is_set():
        status = "loading"
    else:
        status = "ready" if rag else "error"
    return {
        "status": status,
        "error": startup_error,
        "llm": {"configured": bool(llm), "provider": llm.provider if llm else None,
                "models": llm.models if llm else []},
        "tts": {"engines": tts.engines()},
        "characters": list(CHARACTERS),
    }


@app.get("/api/characters", summary="Available characters and their indexes")
def list_characters():
    result = []
    for c in CHARACTERS.values():
        info = rag.index_info(c) if rag else {}
        result.append({"id": c.id, "name": c.name, "title_en": c.title_en, "voice": c.voice,
                       "wikipedia_url": info.get("url"), "chunks": info.get("chunks")})
    return result


@app.post("/api/ask", summary="Ask a historical character a question")
def ask(req: AskRequest):  # sync on purpose: FastAPI runs it in a worker thread
    character = _character(req.character_name)
    engine = _rag()
    try:
        result = engine.ask_question(character, req.question, req.session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("ask failed")
        raise HTTPException(500, f"Server error: {e}")
    return {"success": True, "data": result}


@app.post("/api/reset", summary="Start a new conversation with a character")
def reset(req: SessionRequest):
    _rag().reset_conversation(_character(req.character_name), req.session_id)
    return {"success": True}


@app.get("/api/history", summary="Conversation of one visitor with one character")
def history(character_name: str, session_id: str = "default"):
    if not re.match(SESSION_ID_PATTERN, session_id):
        raise HTTPException(400, "invalid session_id")
    return {"success": True,
            "data": _rag().get_conversation_history(_character(character_name), session_id)}


@app.post("/api/tts", summary="Arabic speech (MP3, base64) with word timings for lip sync")
async def text_to_speech(req: TTSRequest):
    character = resolve_character(req.character_name) if req.character_name else None
    voice = character.voice if character else "ar-SA-HamedNeural"
    try:
        data = dict(await tts.synthesize(req.text, voice))
    except TTSError as e:
        raise HTTPException(503, f"Text-to-speech unavailable: {e}")
    if req.vocalized:
        # same words as spoken, but with their vowel marks: the lip sync then knows the vowels
        data["words"] = vocalize_words(data["words"], req.vocalized)
    return {"success": True, "data": data}


# ---------------------------------------------------------------- website
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(FRONTEND_DIR / "assets" / "img" / "logo.svg", media_type="image/svg+xml")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
