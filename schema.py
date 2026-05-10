"""
Schema definitions for the Incede contact bot.
Contains TypedDict state definitions and Pydantic models for structured outputs.
"""

from typing import Optional, Annotated, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator
from langgraph.graph.message import add_messages


def coerce_bool(v: Any) -> bool:
    """Coerce string booleans to actual bools.
    
    Groq models sometimes return 'true'/'false' as strings instead of
    JSON booleans. This validator normalises them.
    """
    if isinstance(v, str):
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        raise ValueError(f"Cannot coerce {v!r} to bool")
    return bool(v)


class ContactBotState(TypedDict):
    """
    State definition for the LangGraph contact bot workflow.
    Tracks conversation progress and collected contact details.
    """
    messages: Annotated[list, add_messages]
    session_id: str
    name: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    description: Optional[str]
    current_field: str
    is_valid: bool
    is_complete: bool
    error_message: Optional[str]
    raw_user_input: Optional[str]


class ExtractedName(BaseModel):
    """Structured output model for extracting name from user input."""
    name: Optional[str] = Field(
        default=None,
        description="The extracted full name of the user, or None if not found"
    )
    found: bool = Field(
        description="Whether a name was successfully found in the input"
    )

    @field_validator("found", mode="before")
    @classmethod
    def _coerce_found(cls, v: Any) -> bool:
        return coerce_bool(v)


class ExtractedPhone(BaseModel):
    phone: Optional[str] = Field(
        default=None,
        description="The phone number exactly as the user typed it, including any extra digits, spaces, or characters. Do not modify, truncate, or format it in any way."
    )
    found: bool = Field(
        description="Whether a phone number was successfully found in the input"
    )

    @field_validator("found", mode="before")
    @classmethod
    def _coerce_found(cls, v: Any) -> bool:
        return coerce_bool(v)


class ExtractedEmail(BaseModel):
    """Structured output model for extracting email address from user input."""
    email: Optional[str] = Field(
        default=None,
        description="The extracted email address, or None if not found"
    )
    found: bool = Field(
        description="Whether an email address was successfully found in the input"
    )

    @field_validator("found", mode="before")
    @classmethod
    def _coerce_found(cls, v: Any) -> bool:
        return coerce_bool(v)


class ExtractedDescription(BaseModel):
    """Structured output model for extracting description from user input."""
    description: Optional[str] = Field(
        default=None,
        description="The extracted description or purpose text, or None if the user skipped"
    )
    skipped: bool = Field(
        description="Whether the user chose to skip providing a description"
    )

    @field_validator("skipped", mode="before")
    @classmethod
    def _coerce_skipped(cls, v: Any) -> bool:
        return coerce_bool(v)


class ValidationResult(BaseModel):
    """Structured output model for validation tool results."""
    is_valid: bool = Field(
        description="Whether the value passed validation"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for validation failure if is_valid is False"
    )

    @field_validator("is_valid", mode="before")
    @classmethod
    def _coerce_is_valid(cls, v: Any) -> bool:
        return coerce_bool(v)


class SessionInfo(BaseModel):
    """Model for session creation response."""
    session_id: str
    created_at: str


class ChatMessage(BaseModel):
    """Model for individual chat messages sent to/from the bot."""
    session_id: str
    message: str


class LogEntry(BaseModel):
    """Model representing a single log entry for display."""
    session_id: str
    started_at: str
    ended_at: Optional[str]
    status: str
    contact_collected: bool