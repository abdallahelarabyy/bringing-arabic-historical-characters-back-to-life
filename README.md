# Bringing Arabic Historical Characters Back to Life

Graduation project in AI & Data Science at the Egypt-Japan University of Science and Technology (E-JUST).

Students talk, by text or by voice, with **Salah El-Din Al-Ayyubi** and **Muhammad Ali Pasha**. Each character answers **in Arabic and in character** through a 3D avatar whose lips move with the Arabic speech. The answers are grounded in the character's Arabic Wikipedia article (RAG), and every answer links to the sections it came from.

**Tech stack:** Python · FastAPI · LangChain · FAISS · SILMA Arabic embeddings · Groq (Qwen, GPT-OSS, ALLaM) or Gemini · Edge neural TTS · Three.js / TalkingHead · Docker

## Features

- **Text or voice:** type a question or speak it (Arabic speech recognition in the browser), including a hands-free conversation mode.
- **In character:** first-person answers in simple Modern Standard Arabic; events after the character's death are "after my time".
- **Grounded:** passages are retrieved from Arabic Wikipedia and every answer links to the exact sections used.
- **3D avatar with real Arabic lip sync:** a dedicated Arabic module turns every spoken Arabic word, with its vowels, into mouth shapes (see below). No translation to English.
- **Conversation memory:** follow-up questions work, and each visitor has a separate conversation.
- **Costless:** free Groq API for the answers (no credit card), free Microsoft Edge voices for speech. Without any key, answers are quoted from Wikipedia and marked as such.

## How it works

```
question (typed, or spoken and recognised in the browser as ar-SA)
   │
   ▼
POST /api/ask ── FastAPI (backend/main.py)
   │   backend/services/rag_engine.py
   │   1. Arabic Wikipedia article → cleaned (reference sections removed) → split by section
   │   2. SILMA Arabic embeddings → FAISS index (built once, saved in data/faiss_indices/<character>)
   │   3. top-4 passages for the question (+ the previous question for follow-ups)
   │   4. the chat model (Groq / Gemini) answers in the first person, 2–4 sentences, from those
   │      passages + the last 6 turns; a second short call adds the vowels (tashkeel) for the lip sync
   │   5. the turn is saved in data/conversations/<visitor id>/<character>.json
   ▼
chat bubble with the answer (shown without vowel marks) and its Wikipedia sources
   │
   ▼
POST /api/tts ── Arabic speech + the time of every word
   │   free Microsoft Edge neural voices (sentences synthesised in parallel; ElevenLabs optional);
   │   each spoken word gets its vowels
   │   back from the vowelled answer
   ▼
TalkingHead 3D avatar: plays the audio, and lipsync-ar.mjs turns each Arabic word
into mouth shapes (visemes) at its exact time → the lips follow the speech.
If server speech is unavailable, the browser's own Arabic voice is used.
```

## Arabic lip sync

TalkingHead ships lip sync for English, Finnish and a few other languages, but not Arabic. Following its
guide for new languages ("Appendix C" of the TalkingHead README), `frontend/assets/vendor/talkinghead/lipsync-ar.mjs`
maps Arabic graphemes → phonemes → the 15 Oculus visemes directly, the approach TalkingHead used for Finnish.
Arabic spelling is almost phonemic once the vowels are written, so:

| Arabic | Sound | Mouth shape (viseme) |
| --- | --- | --- |
| ب م | b, m | `PP` lips closed |
| ف | f | `FF` lip to teeth |
| ث ذ ظ | th | `TH` tongue between teeth |
| ت د ط ض / ن ل / ر | t d / n l / r | `DD` / `nn` / `RR` |
| س ص ز / ش ج | s z / sh j | `SS` / `CH` |
| ك ق غ خ | k q gh kh | `kk` |
| ح ع | throat sounds | open jaw (`aa`) |
| َ ا ى / ِ ي / ُ و | a, aa / i, ii / u, uu | `aa` / `I` / `U` (long vowels last longer) |

It also handles tanween (+n), shadda (longer consonant), the article ال with sun letters (الشمس → ash-shams),
hamza seats, taa marbuta, diphthongs (ay, aw), the silent alif of plural verbs, and numbers read as Arabic words.
The answer is written normally (better Arabic), then a second short request adds full tashkeel without changing
the words, so the module knows every short vowel; for text without
vowel marks it restores them with a syllable heuristic (Arabic syllables are CV, CVV or CVC).

## Run it

### With Docker (recommended)

```bash
cp .env.example .env               # Windows: copy .env.example .env  (then add your free Groq key)
docker compose up -d --build       # the first build takes a few minutes
```

Open http://localhost:8000. The image already contains the code, the website, the indexes and the Arabic embedding model, so the container is ready about a minute after it starts. Conversations are stored in a Docker volume and survive restarts.

| Command | What it does |
| --- | --- |
| `docker compose logs -f` | show the logs |
| `docker compose up -d` | apply changes to `.env` (recreates the container with the new settings) |
| `docker compose down` | stop and remove the container (the conversations volume is kept) |

### Without Docker (Python 3.10+)

**Windows:** double-click `run.bat`. The first run creates a Python environment and installs everything.

**Any OS:**

```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env               # then add your free Groq key
python -m uvicorn backend.main:app --port 8000
```

The first start downloads the Arabic embedding model (about 0.5 GB) once. Use Chrome or Edge for voice input.

### Settings (`.env`)

| Setting | Needed? | What it does |
| --- | --- | --- |
| `GROQ_API_KEY` | Recommended (free, no credit card: [console.groq.com/keys](https://console.groq.com/keys)) | In-character answers. Without a key the app still works, but answers are quoted directly from Wikipedia and marked with a notice. |
| `LLM_PROVIDER` | No | `groq` (default), `gemini`, `openai` (any OpenAI-compatible server such as a local [Ollama](https://ollama.com): set `OPENAI_BASE_URL=http://localhost:11434/v1`) or `auto`. |
| `GROQ_MODELS` | No | Models tried in order; when one reaches its free daily limit the next answers. Default `qwen/qwen3.8-27b,openai/gpt-oss-120b,allam-2-7b` (Qwen writes the best vowelled Arabic of the free models; ALLaM is an Arabic-specialised model). |
| `GEMINI_API_KEY`, `GEMINI_MODELS` | No | Use Google Gemini instead (free only if your Google project has the free tier). |
| `LLM_DIACRITICS` | No | `true` (default): answers are written with full tashkeel for accurate lip sync. |
| `TTS_ENGINE` | No | `edge` (default, free), `auto`, `elevenlabs` (paid) or `browser`. |
| `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | No | Use ElevenLabs voices instead of the free Edge voices. |

Keys stay on the server and are never sent to the browser. `.env` is excluded from git and from the Docker image.

## API

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| POST | `/api/ask` | `{character_name, question, session_id}` | `{answer, answer_vocalized, sources[], mode: "llm" or "offline", notice, model}` |
| POST | `/api/tts` | `{text, character_name, vocalized?}` | `{audio (MP3, base64), words[] (vowelled when vocalized is given), wtimes[], wdurations[], engine}` |
| POST | `/api/reset` | `{character_name, session_id}` | starts a new conversation |
| GET | `/api/history` | `?character_name=&session_id=` | the saved conversation |
| GET | `/api/characters` | | characters and index info |
| GET | `/health` | | status, LLM and TTS configuration |

Interactive docs: http://localhost:8000/docs. `character_name` accepts the id (`salah-eldin`, `mohamed-ali`) or the Arabic name.

## Project structure

```
├── backend/                       Python server (FastAPI)
│   ├── main.py                    app, API routes, serves the website
│   ├── config.py                  settings from .env, project paths
│   ├── schemas.py                 request bodies of the API
│   ├── characters.py              the characters (Wikipedia title, persona, voice); add new ones here
│   └── services/
│       ├── rag_engine.py          indexes, retrieval, prompt, conversation memory
│       ├── llm.py                 chat model clients: Groq, Gemini, any OpenAI-compatible server
│       ├── wiki_loader.py         downloads and cleans the Arabic Wikipedia article
│       └── tts.py                 text-to-speech (Edge / ElevenLabs) with word timings
├── frontend/                      website, served by the backend at /
│   ├── index.html, about.html     home and about pages
│   ├── characters/                one page per character (salah-eldin.html, mohamed-ali.html)
│   └── assets/
│       ├── css/                   site.css (home, about), character.css (character pages)
│       ├── js/                    site.js (home, about), character.js (chat, speech, avatar)
│       ├── avatars/               3D avatars (GLB, with face shapes for lip sync)
│       ├── img/                   logo and character portraits
│       └── vendor/talkinghead/    TalkingHead 1.4 + lipsync-ar.mjs (Arabic lip sync, see above)
├── data/
│   ├── faiss_indices/             ready-built indexes, rebuilt automatically if deleted
│   └── conversations/             created at runtime, one folder per visitor (not in git)
├── Dockerfile, docker-compose.yml container deployment
├── .env.example                   settings template, copy to .env
├── LICENSE                        MIT
├── requirements.txt
└── run.bat                        one-click start on Windows
```

### Adding a character

1. Add a `Character(...)` entry in `backend/characters.py`, with the exact Arabic Wikipedia title, a persona, the year of death and an Edge voice such as `ar-EG-ShakirNeural`.
2. Copy `frontend/characters/mohamed-ali.html`, change `window.CHARACTER` and the texts, and add a card for it in `frontend/index.html`.
3. Restart the server. The index is built from Wikipedia automatically.

## Limitations

- The characters are AI simulations built from encyclopedia text; answers can simplify or get details wrong, which is why every answer shows its sources.
- Voice input relies on the browser's speech recognition (Chrome or Edge) and a microphone.
- The Edge voices and the Groq/Gemini APIs need an internet connection; the free Groq tier has daily limits per model (the app moves to the next model when one is used up).

## Credits

- [TalkingHead](https://github.com/met4citizen/TalkingHead) by Mika Suominen (MIT license, see `frontend/assets/vendor/talkinghead/LICENSE`); the Arabic lip-sync module `lipsync-ar.mjs` was written for this project.
- Historical content from [Arabic Wikipedia](https://ar.wikipedia.org), available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- Arabic embeddings: [silma-ai/silma-embedding-matryoshka-v0.1](https://huggingface.co/silma-ai/silma-embedding-matryoshka-v0.1).
- Speech: Microsoft Edge neural voices through [edge-tts](https://github.com/rany2/edge-tts).

## License

The code of this project is released under the [MIT License](LICENSE). Third-party parts keep their own licenses: TalkingHead (MIT), and the Wikipedia text inside `data/faiss_indices/` (CC BY-SA 4.0).
