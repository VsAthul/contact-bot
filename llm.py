"""
LLM configuration module for the Incede contact bot.
Sets up Groq-based language model clients for chat, structured output, and tool calling.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Model identifier for Groq API
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# System prompt that defines the bot's persona and behaviour for all generated replies.
# Every call to generate_response() is governed by these rules.
BOT_SYSTEM_PROMPT = """You are a friendly and professional contact-collection assistant for Incede, \
a software company. Your only job is to collect the user's contact details — \
name, phone number, email address, and an optional description — so that the \
Incede team can get in touch with them.

Rules you must follow at all times:
- Keep every reply concise (1–3 sentences).
- Be warm, polite, and encouraging.
- Never make up or guess any contact details.
- Never answer questions unrelated to collecting contact information; \
  instead gently redirect the user back to providing their details.
- When asking for a field, be specific about what format you expect \
  (e.g. for phone: a 10-digit Indian mobile number starting with 6, 7, 8, or 9).
- When a value is invalid, briefly explain what was wrong and ask again clearly.
- When a value is accepted, acknowledge it warmly before moving on.
- When all details are collected, thank the user and confirm the Incede \
  team will be in touch soon.
- Do NOT output JSON, bullet points, or markdown — plain conversational text only.

- When the user provides a phone number, pass it EXACTLY as given to the validate_phone tool — 
do NOT extract, truncate, reformat, or modify the input in any way before calling the tool.
"""


def get_llm() -> ChatGroq:
    """
    Create and return a base ChatGroq LLM instance.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set")

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0.7,   # higher temp for natural-sounding conversational replies
        max_tokens=1256,
    )


def get_structured_llm(schema):
    """
    Create a ChatGroq LLM instance bound to a structured output schema.
    Uses temperature=0 for precise, deterministic extraction.

    Args:
        schema: A Pydantic BaseModel class to use for structured output parsing.

    Returns:
        A ChatGroq instance configured to return structured output matching the schema.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=256,
    )
    return llm.with_structured_output(schema, method="json_mode")


def get_tool_llm(tools: list) -> ChatGroq:
    """
    Create a ChatGroq LLM instance bound with tool calling capabilities.

    Args:
        tools: A list of LangChain tool objects to bind to the LLM.

    Returns:
        A ChatGroq instance configured with the provided tools.
    """
    llm = get_llm()
    return llm.bind_tools(tools)


# ── NEW ───────────────────────────────────────────────────────────────────────
from typing import Optional


def get_validation_llm(tools: list, tool_name: Optional[str] = None) -> ChatGroq:
    """
    Create a ChatGroq LLM instance bound to validation tools.

    Args:
        tools: List containing ONLY the required validation tool.
        tool_name: Exact tool name to force during tool calling.

    Returns:
        ChatGroq instance configured for deterministic validation.
    """

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set")
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=256,
    )

    bind_kwargs = {}

    if tool_name:
        bind_kwargs["tool_choice"] = {
            "type": "function",
            "function": {
                "name": tool_name
            }
        }

    return llm.bind_tools(tools, **bind_kwargs)
# ── END NEW ───────────────────────────────────────────────────────────────────


def generate_response(instruction: str) -> str:
    """
    Generate a conversational reply from the LLM using the bot's system prompt.

    The caller describes *what the bot needs to say* via `instruction` —
    including context such as which field is being collected, whether the
    previous value was valid or invalid, and the reason for any failure.
    The LLM then composes a natural, on-brand reply.

    Args:
        instruction: Plain-English description of what the bot should say,
                     with all relevant context embedded.

    Returns:
        The generated reply string to send to the user.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=BOT_SYSTEM_PROMPT),
        HumanMessage(content=instruction),
    ]
    response = llm.invoke(messages)
    return response.content.strip()