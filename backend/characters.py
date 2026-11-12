"""The historical characters the app supports.

To add a character: add an entry here, create its page in frontend/characters/
(copy an existing page and change window.CHARACTER), then start the server.
Its Wikipedia index is built automatically on first start.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Character:
    id: str            # URL-safe id, also the folder name of its FAISS index
    name: str          # Arabic display name (also accepted by the API)
    wiki_title: str    # exact Arabic Wikipedia article title
    title_en: str
    persona: str       # who the model plays, in Arabic
    death_year: int    # events after this year happened "after my time"
    voice: str         # Microsoft Edge neural voice for text-to-speech


CHARACTERS = {
    c.id: c
    for c in [
        Character(
            id="salah-eldin",
            name="صلاح الدين الأيوبي",
            wiki_title="صلاح الدين الأيوبي",
            title_en="Salah El-Din Al-Ayyubi",
            persona=(
                "أنت السلطان الناصر صلاح الدين يوسف بن أيوب، مؤسس الدولة الأيوبية "
                "وسلطان مصر والشام (532–589 هـ / 1137–1193 م)، ومحرر بيت المقدس عام 1187م."
            ),
            death_year=1193,
            voice="ar-SA-HamedNeural",
        ),
        Character(
            id="mohamed-ali",
            name="محمد علي باشا",
            wiki_title="محمد علي باشا",
            title_en="Muhammad Ali Pasha",
            persona=(
                "أنت محمد علي باشا، والي مصر ومؤسس الأسرة العلوية (1769–1849م)، "
                "ومؤسس الدولة المصرية الحديثة بجيشها ومدارسها ومصانعها."
            ),
            death_year=1849,
            voice="ar-EG-ShakirNeural",
        ),
    ]
}


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def resolve_character(key: Optional[str]) -> Optional[Character]:
    """Find a character by id (salah-eldin) or Arabic name (صلاح الدين الأيوبي)."""
    if not key:
        return None
    key = _normalize(key)
    if key.lower() in CHARACTERS:
        return CHARACTERS[key.lower()]
    for c in CHARACTERS.values():
        if key in (c.name, c.wiki_title, c.title_en):
            return c
    return None
