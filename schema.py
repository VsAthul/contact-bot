"""
Schema definitions for the Incede contact bot.
Contains TypedDict state definitions and Pydantic models for structured outputs.
"""

from typing import Optional, Annotated, Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator
from langgraph.graph.message import add_messages


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def coerce_bool(v: Any) -> bool:
    """
    Coerce string booleans to actual bools.

    Some LLM providers occasionally return:
    - "true"
    - "false"

    as strings instead of JSON booleans.
    """

    if isinstance(v, str):

        lowered = v.strip().lower()

        if lowered == "true" or lowered == "yes" or lowered == "1":
            return True

        if lowered == "false" or lowered == "no" or lowered == "0":
            return False
        else :

            raise ValueError(f"Cannot coerce {v!r} to bool")

    return bool(v)


VALID_INTENTS = Literal[
    "answer",
    "question",
    "off_topic",
    "unclear",
]


# --------------------------------------------------------------------------
# LangGraph state
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Structured extraction models
# --------------------------------------------------------------------------

class ExtractedName(BaseModel):
    """
    Structured output model for extracting a user's name.
    """

    name: Optional[str] = Field(
        default=None,
        description=(
            "The extracted name from the user's message. "
            "Can be a first name or full name."
        ),
    )

    found: bool = Field(
        description=(
            "Whether a valid name was found in the user's input."
        )
    )

    intent: VALID_INTENTS = Field(
        default="answer",
        description=(
            "Intent classification for the user's message. "
            "Must be one of: "
            "'answer', 'question', 'off_topic', or 'unclear'."
        ),
    )

    @field_validator("found", mode="before")
    @classmethod
    def _coerce_found(cls, v: Any) -> bool:
        return coerce_bool(v)

    @field_validator("intent", mode="before")
    @classmethod
    def _coerce_intent(cls, v: Any) -> str:

        if v is None:
            return "answer"

        if not isinstance(v, str):
            return "unclear"

        normalized = v.strip().lower()

        allowed = {
            "answer",
            "question",
            "off_topic",
            "unclear",
        }

        if normalized not in allowed:
            return "unclear"

        return normalized


class ExtractedPhone(BaseModel):
    """
    Structured output model for extracting phone number.
    """

    phone: Optional[str] = Field(
        default=None,
        description=(
            "The phone number exactly as the user typed it, "
            "including spaces, country codes, punctuation, "
            "or invalid characters. "
            "Do not modify or normalize it."
        ),
    )

    found: bool = Field(
        description=(
            "Whether a phone number was found in the user's input."
        )
    )

    intent: VALID_INTENTS = Field(
        default="answer",
        description=(
            "Intent classification for the user's message."
        ),
    )

    @field_validator("found", mode="before")
    @classmethod
    def _coerce_found(cls, v: Any) -> bool:
        return coerce_bool(v)

    @field_validator("intent", mode="before")
    @classmethod
    def _coerce_intent(cls, v: Any) -> str:

        if v is None:
            return "answer"

        if not isinstance(v, str):
            return "unclear"

        normalized = v.strip().lower()

        allowed = {
            "answer",
            "question",
            "off_topic",
            "unclear",
        }

        if normalized not in allowed:
            return "unclear"

        return normalized


class ExtractedEmail(BaseModel):
    """
    Structured output model for extracting email address.
    """

    email: Optional[str] = Field(
        default=None,
        description=(
            "The extracted email address from the user's input."
        ),
    )

    found: bool = Field(
        description=(
            "Whether an email address was found in the input."
        )
    )

    intent: VALID_INTENTS = Field(
        default="answer",
        description=(
            "Intent classification for the user's message."
        ),
    )

    @field_validator("found", mode="before")
    @classmethod
    def _coerce_found(cls, v: Any) -> bool:
        return coerce_bool(v)

    @field_validator("intent", mode="before")
    @classmethod
    def _coerce_intent(cls, v: Any) -> str:

        if v is None:
            return "answer"

        if not isinstance(v, str):
            return "unclear"

        normalized = v.strip().lower()

        allowed = {
            "answer",
            "question",
            "off_topic",
            "unclear",
        }

        if normalized not in allowed:
            return "unclear"

        return normalized


class ExtractedDescription(BaseModel):
    """
    Structured output model for extracting description or message.
    """

    description: Optional[str] = Field(
        default=None,
        description=(
            "The extracted description or message text "
            "provided by the user."
        ),
    )

    skipped: bool = Field(
        description=(
            "Whether the user intentionally skipped "
            "providing a description."
        )
    )

    found: bool = Field(
        default=True,
        description=(
            "Whether a usable description/message was found."
        ),
    )

    intent: VALID_INTENTS = Field(
        default="answer",
        description=(
            "Intent classification for the user's message."
        ),
    )

    @field_validator("skipped", mode="before")
    @classmethod
    def _coerce_skipped(cls, v: Any) -> bool:
        return coerce_bool(v)

    @field_validator("found", mode="before")
    @classmethod
    def _coerce_found(cls, v: Any) -> bool:
        return coerce_bool(v)

    @field_validator("intent", mode="before")
    @classmethod
    def _coerce_intent(cls, v: Any) -> str:

        if v is None:
            return "answer"

        if not isinstance(v, str):
            return "unclear"

        normalized = v.strip().lower()

        allowed = {
            "answer",
            "question",
            "off_topic",
            "unclear",
        }

        if normalized not in allowed:
            return "unclear"

        return normalized


# --------------------------------------------------------------------------
# Validation result models
# --------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """
    Structured output model for validation tool results.
    """

    is_valid: bool = Field(
        description="Whether the value passed validation."
    )

    reason: Optional[str] = Field(
        default=None,
        description=(
            "Reason for validation failure if is_valid is False."
        ),
    )

    @field_validator("is_valid", mode="before")
    @classmethod
    def _coerce_is_valid(cls, v: Any) -> bool:
        return coerce_bool(v)


# --------------------------------------------------------------------------
# API / session models
# --------------------------------------------------------------------------

class SessionInfo(BaseModel):
    """
    Model for session creation response.
    """

    session_id: str
    created_at: str


class ChatMessage(BaseModel):
    """
    Model for individual chat messages sent to/from the bot.
    """

    session_id: str
    message: str


class LogEntry(BaseModel):
    """
    Model representing a single log entry for display.
    """

    session_id: str
    started_at: str
    ended_at: Optional[str]
    status: str
    contact_collected: bool