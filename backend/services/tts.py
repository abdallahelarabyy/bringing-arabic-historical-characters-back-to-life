"""Arabic text-to-speech with word timings for the avatar's lip sync.

Engines, tried in order (TTS_ENGINE=auto):
  1. ElevenLabs  - only when ELEVENLABS_API_KEY is set (paid, most natural voice)
  2. Edge TTS    - free Microsoft neural voices, no key needed (needs internet)
If both fail the API returns 503 and the page falls back to the browser's own speech.

The key stays on the server: the browser only calls /api/tts.
"""
import asyncio
import base64
import logging
import re
from collections import OrderedDict
from typing import Dict, List, Optional

import requests

log = logging.getLogger("tts")

ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps"


class TTSError(RuntimeError):
    pass


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*#_`>\[\]{}|~^]+", " ", text)
    text = re.sub(r"[^\w\s؀-ۿ.,!?؟،؛:()\-'\"«»]", " ", text)
    return " ".join(text.split())[:1500]


_MARKS = re.compile(r"[ً-ْٰ]")


def _letters(word: str, keep_marks: bool = False) -> str:
    return "".join(ch for ch in word if ch.isalnum() or (keep_marks and _MARKS.match(ch)))


def vocalize_words(words: List[str], vocalized: str) -> List[str]:
    """Give each spoken word its vowel marks (tashkeel) from the vocalized version of the text.

    The voice engine reads the plain text and reports its words; the lip sync is far more accurate
    with the vowels, so every word is matched (ignoring marks and punctuation) to the same word in
    the vocalized text. Words that cannot be matched are kept as they are.
    """
    tokens = vocalized.split()
    keys = [_letters(_MARKS.sub("", t)) for t in tokens]
    out, j = [], 0
    for w in words:
        key = _letters(_MARKS.sub("", w))
        replacement = w
        for k in range(j, min(j + 4, len(tokens))):
            if key and keys[k] == key:
                replacement = _letters(tokens[k], keep_marks=True)
                j = k + 1
                break
        out.append(replacement)
    return out


async def _edge(text: str, voice: str, rate: str) -> Dict:
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    audio = bytearray()
    words: List[str] = []
    wtimes: List[int] = []
    wdurations: List[int] = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            words.append(chunk["text"])
            wtimes.append(chunk["offset"] // 10_000)        # 100 ns units -> ms
            wdurations.append(chunk["duration"] // 10_000)
    if not audio:
        raise TTSError("Edge TTS returned no audio")
    return {"audio": bytes(audio), "words": words, "wtimes": wtimes, "wdurations": wdurations}


# Edge's default output is 48 kbit/s constant-bitrate MP3: 6 bytes per millisecond of audio
EDGE_BYTES_PER_MS = 48_000 / 8 / 1000
SENTENCE_RE = re.compile(r"[^.!?؟؛\n]+[.!?؟؛]*")


def split_sentences(text: str, max_len: int = 180) -> List[str]:
    """Split into sentences (long ones at the Arabic comma) so they can be synthesised in parallel."""
    parts: List[str] = []
    for s in (m.group().strip() for m in SENTENCE_RE.finditer(text)):
        while len(s) > max_len:
            cut = max(s.rfind("،", 0, max_len), s.rfind(",", 0, max_len), s.rfind(" ", 0, max_len))
            if cut <= 0:
                cut = max_len
            parts.append(s[: cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            parts.append(s)
    return parts or [text]


async def _edge_one(text: str, voice: str, rate: str) -> Dict:
    try:
        return await asyncio.wait_for(_edge(text, voice, rate), timeout=30)
    except Exception as first:  # the Edge service occasionally drops a request
        log.info("Edge TTS retry after: %s", first)
        return await asyncio.wait_for(_edge(text, voice, rate), timeout=30)


async def _edge_parallel(text: str, voice: str, rate: str) -> Dict:
    """Edge is slow on long texts but fast on short ones: synthesise every sentence at the same
    time, then join the MP3s and shift each sentence's word timings by the audio before it."""
    results = await asyncio.gather(*[_edge_one(s, voice, rate) for s in split_sentences(text)])
    audio = bytearray()
    words: List[str] = []
    wtimes: List[int] = []
    wdurations: List[int] = []
    for r in results:
        offset = round(len(audio) / EDGE_BYTES_PER_MS)
        audio += r["audio"]
        words += r["words"]
        wtimes += [t + offset for t in r["wtimes"]]
        wdurations += r["wdurations"]
    return {"audio": bytes(audio), "words": words, "wtimes": wtimes, "wdurations": wdurations}


def _eleven(text: str, api_key: str, voice: str, model: str) -> Dict:
    r = requests.post(
        ELEVEN_URL.format(voice=voice), timeout=60,
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"text": text, "model_id": model,
              "voice_settings": {"stability": 0.5, "similarity_boost": 0.6}},
    )
    if r.status_code != 200:
        raise TTSError(f"ElevenLabs HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    al = data.get("alignment") or {}
    chars = al.get("characters", [])
    starts = al.get("character_start_times_seconds", [])
    ends = al.get("character_end_times_seconds", [])
    words, wtimes, wdurations = [], [], []
    cur, cur_start, cur_end = "", 0.0, 0.0
    for ch, s, e in zip(chars, starts, ends):
        if ch.isspace():
            if cur:
                words.append(cur); wtimes.append(int(cur_start * 1000))
                wdurations.append(max(1, int((cur_end - cur_start) * 1000)))
            cur = ""
            continue
        if not cur:
            cur_start = s
        cur += ch
        cur_end = e
    if cur:
        words.append(cur); wtimes.append(int(cur_start * 1000))
        wdurations.append(max(1, int((cur_end - cur_start) * 1000)))
    return {"audio": base64.b64decode(data["audio_base64"]),
            "words": words, "wtimes": wtimes, "wdurations": wdurations}


class TTSService:
    def __init__(self, engine: str = "auto", elevenlabs_key: Optional[str] = None,
                 elevenlabs_voice: str = "", elevenlabs_model: str = "eleven_multilingual_v2",
                 edge_rate: str = "-5%", cache_size: int = 64):
        self.engine = engine.lower()
        self.eleven_key = elevenlabs_key
        self.eleven_voice = elevenlabs_voice
        self.eleven_model = elevenlabs_model
        self.edge_rate = edge_rate
        self._cache: "OrderedDict[tuple, Dict]" = OrderedDict()
        self._cache_size = cache_size

    def engines(self) -> List[str]:
        if self.engine == "browser":
            return []
        order = []
        if self.engine in ("auto", "elevenlabs") and self.eleven_key and self.eleven_voice:
            order.append("elevenlabs")
        if self.engine in ("auto", "edge"):
            order.append("edge")
        return order

    async def synthesize(self, text: str, edge_voice: str) -> Dict:
        text = clean_for_speech(text)
        if not text:
            raise TTSError("empty text")
        errors = []
        for engine in self.engines():
            key = (engine, edge_voice, text)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            try:
                if engine == "elevenlabs":
                    res = await asyncio.to_thread(_eleven, text, self.eleven_key,
                                                  self.eleven_voice, self.eleven_model)
                else:
                    res = await _edge_parallel(text, edge_voice, self.edge_rate)
                out = {
                    "engine": engine, "mime": "audio/mpeg",
                    "audio": base64.b64encode(res["audio"]).decode(),
                    "words": res["words"], "wtimes": res["wtimes"], "wdurations": res["wdurations"],
                }
                self._cache[key] = out
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                return out
            except Exception as e:
                log.warning("TTS engine %s failed: %s", engine, e)
                errors.append(f"{engine}: {e}")
        raise TTSError("; ".join(errors) or "server TTS disabled (TTS_ENGINE=browser)")
