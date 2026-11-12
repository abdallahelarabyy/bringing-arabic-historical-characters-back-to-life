"""Request bodies accepted by the API."""
from typing import Optional

from pydantic import BaseModel, Field

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class AskRequest(BaseModel):
    character_name: str = Field(..., description="Character id (salah-eldin) or Arabic name")
    question: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field("default", pattern=SESSION_ID_PATTERN,
                            description="Visitor id; each visitor has a separate conversation")


class SessionRequest(BaseModel):
    character_name: str
    session_id: str = Field("default", pattern=SESSION_ID_PATTERN)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500)
    character_name: Optional[str] = None
    vocalized: Optional[str] = Field(None, max_length=4000,
                                     description="Same text with tashkeel, used only for the lip sync")
