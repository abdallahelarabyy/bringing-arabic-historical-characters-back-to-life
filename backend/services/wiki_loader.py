"""Download an Arabic Wikipedia article as clean plain-text sections.

Uses the MediaWiki API directly (with a proper User-Agent, which Wikipedia requires;
the old `wikipedia` package was getting HTTP 429) and drops the reference, notes and
external-link sections, which are not useful for answering questions.
"""
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import requests

API = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "ArabicHistoricalCharacters/2.0 (graduation project; educational RAG chatbot) python-requests"

# Headings of sections with no narrative content (matched as substrings after normalizing)
SKIP_KEYWORDS = ("مراجع", "مصادر", "هوامش", "ملاحظات", "وصلات خارجيه", "روابط خارجيه", "انظر ايضا",
                 "اقرا ايضا", "قراءات اضافيه", "قراءه اضافيه", "ببليوغرافيا", "معرض صور", "معرض الصور")
HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


def _norm(s: str) -> str:
    s = re.sub(r"[ً-ْـ]", "", s)  # diacritics and tatweel
    s = re.sub("[أإآ]", "ا", s).replace("ة", "ه").replace("ى", "ي")
    return " ".join(s.split())


@dataclass
class Article:
    title: str
    url: str
    revision: int
    sections: List[Tuple[str, str]] = field(default_factory=list)  # (heading path, text)

    @property
    def text(self) -> str:
        return "\n\n".join(t for _, t in self.sections)


def fetch_article(title: str, lang: str = "ar", retries: int = 4, timeout: int = 30) -> Article:
    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "extracts|revisions|info", "explaintext": 1, "exsectionformat": "wiki",
        "rvprop": "ids", "inprop": "url", "titles": title,
    }
    delay = 2.0
    for attempt in range(retries):
        try:
            r = requests.get(API.format(lang=lang), params=params, timeout=timeout,
                             headers={"User-Agent": USER_AGENT})
            if r.status_code in (429, 503):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            page = r.json()["query"]["pages"][0]
            if page.get("missing"):
                raise ValueError(f"Wikipedia article not found: {title}")
            return _parse(page)
        except ValueError:
            raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _parse(page: dict) -> Article:
    art = Article(title=page["title"], url=page.get("fullurl", ""),
                  revision=(page.get("revisions") or [{}])[0].get("revid", 0))
    path: List[str] = []           # current heading path, e.g. ["حياته", "نشأته"]
    skip_level = None              # inside a skipped section until a heading of this level or higher
    buf: List[str] = []

    def flush():
        text = "\n".join(buf).strip()
        if text and skip_level is None:
            art.sections.append((" › ".join(path) or "مقدمة", text))
        buf.clear()

    for line in page.get("extract", "").splitlines():
        m = HEADING.match(line.strip())
        if not m:
            buf.append(line)
            continue
        flush()
        level, heading = len(m.group(1)), m.group(2).strip()
        if skip_level is not None and level <= skip_level:
            skip_level = None
        del path[level - 2:]
        path.append(heading)
        if skip_level is None and any(k in _norm(heading) for k in SKIP_KEYWORDS):
            skip_level = level
    flush()
    return art
