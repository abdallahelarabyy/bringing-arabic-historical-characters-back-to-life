"""Arabic Wikipedia RAG engine that answers *as* a historical character.

Pipeline for one question:
  1. the character's Arabic Wikipedia article (cleaned, split by section) is embedded with
     SILMA Arabic embeddings and stored in a FAISS index (built once, then loaded from disk);
  2. the most relevant passages are retrieved for the question (plus the previous question,
     so follow-ups such as "and how did you do it?" still find the right passages);
  3. the chat model writes a short first-person answer from those passages and the recent
     conversation, fully vowelled (tashkeel) so the avatar's lips can follow the real vowels;
  4. the turn is saved in a per-visitor conversation file.

If no chat model is configured or the call fails, an "offline" answer is built from the best
passage so the app (voice, avatar, sources) still works.
"""
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.characters import CHARACTERS, Character
from backend.services.llm import ChatClient, LLMError
from backend.services.wiki_loader import fetch_article

log = logging.getLogger("rag")

EMBEDDING_MODEL = "silma-ai/silma-embedding-matryoshka-v0.1"
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DIACRITICS_RE = re.compile(r"[ً-ْٰ]")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

SYSTEM_PROMPT = """{persona}
تتحدث الآن مع طالب يريد أن يتعلم التاريخ منك مباشرة.

قواعد الإجابة:
- تكلم بصيغة المتكلم دائمًا وكأنك أنت الشخصية نفسها، ولا تخرج عن هذا الدور أبدًا، ولا تذكر أنك نموذج ذكاء اصطناعي.
- أجب باللغة العربية الفصحى السهلة، في جملتين إلى أربع جمل فقط، لأن إجابتك ستُقرأ بصوت مسموع.
- لا تستخدم التنسيق (لا عناوين ولا نقاط ولا رموز مثل * أو #) ولا الرموز التعبيرية، واكتب الأرقام بالأرقام العادية.
- اعتمد على "المعلومات التاريخية" أدناه، ولا تذكر أنها نصوص أو مصادر مقدمة لك.
- إذا لم تجد الإجابة في المعلومات ولم تكن متأكدًا منها، فقل بلطف وبأسلوب الشخصية إنك لا تذكر ذلك، ولا تختلق أحداثًا أو أرقامًا.
- إذا سُئلت عن أحداث بعد وفاتك عام {death_year}م، فقل إنها وقعت بعد زمانك.
- إذا كان السؤال خارج التاريخ تمامًا، فأعد الحديث بلطف إلى حياتك وزمانك.
- اكتب بلغة عربية سليمة نحويًا وإملائيًا.

المعلومات التاريخية:
{context}"""

# Second, separate step: add the vowels without changing a letter. Asking for tashkeel while the
# answer is being written makes free models write worse Arabic, so the two jobs are kept apart.
VOCALIZE_PROMPT = """أنت مدقق لغوي متخصص في تشكيل النصوص العربية.
أعد كتابة النص الذي يرسله المستخدم كما هو حرفًا بحرف، مع إضافة التشكيل الكامل (الفتحة والضمة والكسرة والسكون والشدة والتنوين) على كل كلمة.
لا تغيّر أي كلمة، ولا تصحح شيئًا، ولا تضف ولا تحذف أي حرف أو علامة ترقيم، وأخرج النص المشكول فقط دون أي تعليق."""
LETTERS_RE = re.compile(r"[^ء-ي0-9٠-٩]")


class LLMUnavailable(RuntimeError):
    """No chat model is configured or it returned nothing."""


def strip_diacritics(text: str) -> str:
    return DIACRITICS_RE.sub("", text)


class ArabicWikiRAG:
    def __init__(
        self,
        index_dir: Union[str, os.PathLike],
        conversations_dir: Union[str, os.PathLike],
        llm: Optional[ChatClient] = None,
        diacritics: bool = True,
        top_k: int = 4,
        history_turns: int = 6,
    ):
        self.index_dir = os.fspath(index_dir)
        self.history_dir = os.fspath(conversations_dir)
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        self.llm = llm
        self.diacritics = diacritics
        self.top_k = top_k
        self.history_turns = history_turns

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "، ", " ", ""],
        )

        self._stores: Dict[str, FAISS] = {}
        self._index_lock = threading.Lock()
        self._session_locks: Dict[str, threading.Lock] = {}
        self._session_locks_guard = threading.Lock()

    # ------------------------------------------------------------------ indexes
    def _index_path(self, character: Character) -> str:
        return os.path.join(self.index_dir, character.id)

    def build_index(self, character: Character) -> FAISS:
        """Download the article, split it by section and build a FAISS index on disk."""
        log.info("Building index for %s from Wikipedia ...", character.name)
        article = fetch_article(character.wiki_title)
        docs: List[Document] = []
        for section, text in article.sections:
            for chunk in self.splitter.split_text(text):
                docs.append(Document(
                    # the section path is embedded with the text: it helps retrieval a lot
                    page_content=f"{section}\n{chunk}",
                    metadata={"title": article.title, "section": section,
                              "url": article.url, "chunk": len(docs)},
                ))
        store = FAISS.from_documents(docs, self.embeddings)
        path = self._index_path(character)
        store.save_local(path)
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "character": character.name, "wikipedia_title": article.title, "url": article.url,
                "revision": article.revision, "chunks": len(docs), "embedding_model": EMBEDDING_MODEL,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, f, ensure_ascii=False, indent=2)
        log.info("Index for %s: %d chunks", character.name, len(docs))
        return store

    def get_store(self, character: Character) -> FAISS:
        if character.id in self._stores:
            return self._stores[character.id]
        with self._index_lock:
            if character.id not in self._stores:
                path = self._index_path(character)
                if os.path.exists(os.path.join(path, "index.faiss")):
                    store = FAISS.load_local(path, self.embeddings,
                                             allow_dangerous_deserialization=True)
                else:
                    store = self.build_index(character)
                self._stores[character.id] = store
        return self._stores[character.id]

    def index_info(self, character: Character) -> dict:
        try:
            with open(os.path.join(self._index_path(character), "meta.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def warmup(self) -> None:
        """Load the embedding model and every character's index (building missing ones)."""
        for c in CHARACTERS.values():
            try:
                self.get_store(c)
            except Exception as e:  # keep the server usable even if one index fails
                log.error("Could not load index for %s: %s", c.name, e)

    # ------------------------------------------------------------------ history
    def _history_path(self, character: Character, session_id: str) -> str:
        if not SESSION_RE.match(session_id):
            raise ValueError("invalid session_id")
        return os.path.join(self.history_dir, session_id, f"{character.id}.json")

    def _session_lock(self, key: str) -> threading.Lock:
        with self._session_locks_guard:
            return self._session_locks.setdefault(key, threading.Lock())

    def get_conversation_history(self, character: Character, session_id: str) -> List[dict]:
        path = self._history_path(character, session_id)
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return []

    def _save_history(self, character: Character, session_id: str, messages: List[dict]) -> None:
        path = self._history_path(character, session_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(messages[-100:], f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def reset_conversation(self, character: Character, session_id: str) -> None:
        with self._session_lock(f"{session_id}/{character.id}"):
            path = self._history_path(character, session_id)
            if os.path.exists(path):
                os.remove(path)

    # ------------------------------------------------------------------ answering
    def retrieve(self, character: Character, query: str, k: Optional[int] = None) -> List[Document]:
        return self.get_store(character).similarity_search(query, k=k or self.top_k)

    @staticmethod
    def _offline_answer(docs: List[Document]) -> str:
        if not docs:
            return "لا أذكر شيئًا عن هذا الأمر."
        text = docs[0].page_content.split("\n", 1)[-1]
        sentences = [s.strip(" .،,") for s in re.split(r"(?<=[.!؟?])\s+|\n+", text)]
        sentences = [s for s in sentences if len(s) > 20]
        answer = ""
        for s in sentences:
            if answer and len(answer) + len(s) > 350:
                break
            answer = f"{answer} {s}.".strip()
        if len(answer) > 400:  # one very long sentence: cut at a word boundary
            answer = answer[:380].rsplit(" ", 1)[0] + "…"
        return answer or text[:350]

    def _messages(self, character: Character, context: str, recent: List[dict], question: str) -> List[dict]:
        system = SYSTEM_PROMPT.format(persona=character.persona, death_year=character.death_year, context=context)
        messages = [{"role": "system", "content": system}]
        for m in recent:
            # earlier answers are sent without tashkeel: same meaning, far fewer tokens
            content = strip_diacritics(m["content"]) if m["role"] == "assistant" else m["content"]
            messages.append({"role": "user" if m["role"] == "user" else "assistant", "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    def vocalize(self, text: str) -> str:
        """Return `text` with full tashkeel, or `text` unchanged if the model altered any letter."""
        if not (self.diacritics and self.llm and text):
            return text
        try:
            out = self.llm.chat([{"role": "system", "content": VOCALIZE_PROMPT},
                                 {"role": "user", "content": text}], temperature=0)
        except Exception as e:
            log.info("vocalization skipped: %s", str(e)[:200])
            return text
        out = " ".join(THINK_RE.sub("", out).replace("ـ", "").split())
        # Words are later matched one by one (tts.vocalize_words), so a few changed letters are fine;
        # a rewritten sentence is not.
        ratio = SequenceMatcher(None, LETTERS_RE.sub("", strip_diacritics(out)), LETTERS_RE.sub("", text)).ratio()
        if ratio < 0.8:
            log.info("vocalization changed the text (similarity %.2f), keeping the plain answer", ratio)
            return text
        return out

    def ask_question(self, character: Character, question: str, session_id: str = "default") -> Dict:
        question = " ".join(question.split())
        with self._session_lock(f"{session_id}/{character.id}"):
            history = self.get_conversation_history(character, session_id)
            recent = history[-2 * self.history_turns:]

            # The question alone ranks first; for short follow-ups ("how did you do that?"), one
            # passage found with the previous question added is appended so they keep their context.
            docs = self.retrieve(character, question)
            last_user = next((m["content"] for m in reversed(recent) if m["role"] == "user"), "")
            if last_user and len(question) < 60:
                seen = {d.metadata.get("chunk") for d in docs}
                extra = [d for d in self.retrieve(character, f"{last_user}\n{question}", k=2)
                         if d.metadata.get("chunk") not in seen]
                docs = docs[: self.top_k - len(extra[:1])] + extra[:1]
            context = "\n\n---\n\n".join(d.page_content for d in docs)

            mode, notice, model = "llm", None, None
            try:
                if self.llm is None:
                    raise LLMUnavailable("no chat model configured")
                raw = self.llm.chat(self._messages(character, context, recent, question))
                answer = re.sub(r"[*#_`]+", "", THINK_RE.sub("", raw)).strip()  # spoken: no markdown
                answer = strip_diacritics(" ".join(answer.replace("ـ", "").split()))  # no tatweel
                if not answer:
                    raise LLMUnavailable("empty answer")
                model = self.llm.model
                vocalized = self.vocalize(answer)
            except Exception as e:
                log.warning("LLM unavailable, using offline answer: %s", str(e)[:300])
                mode = "offline"
                notice = _explain_llm_error(e)
                answer = vocalized = self._offline_answer(docs)

            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            history += [{"role": "user", "content": question, "time": now},
                        {"role": "assistant", "content": vocalized, "time": now, "mode": mode}]
            self._save_history(character, session_id, history)

        sources, seen = [], set()
        for d in docs:
            key = d.metadata.get("section")
            if key in seen:
                continue
            seen.add(key)
            sources.append({"title": d.metadata.get("title", character.wiki_title),
                            "section": key, "url": d.metadata.get("url", "")})
        return {"answer": answer, "answer_vocalized": vocalized, "sources": sources,
                "mode": mode, "notice": notice, "model": model}


def _explain_llm_error(e: Exception) -> str:
    tail = "، لذلك هذه الإجابة مقتطفة مباشرة من ويكيبيديا."
    status = getattr(e, "status", None)
    msg = str(e)
    if isinstance(e, LLMUnavailable) and "configured" in msg:
        return "لم يتم ضبط مفتاح نموذج الذكاء الاصطناعي (GROQ_API_KEY في ملف .env)" + tail
    if status == 429 or "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "تم تجاوز حد الاستخدام المجاني لنموذج الذكاء الاصطناعي مؤقتًا" + tail
    if status in (401, 403) or "API key" in msg or "PERMISSION_DENIED" in msg:
        return "مفتاح نموذج الذكاء الاصطناعي غير صالح أو غير مسموح له بالاستخدام المجاني" + tail
    if status == 404:
        return "اسم نموذج الذكاء الاصطناعي غير متاح، راجع إعدادات النموذج في ملف .env" + tail
    if isinstance(e, LLMError):
        return "تعذر الاتصال بنموذج الذكاء الاصطناعي" + tail
    return "تعذر الحصول على إجابة من نموذج الذكاء الاصطناعي" + tail
