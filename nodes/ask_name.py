"""
Node: ask_name
Prompts the user for their name.
"""

from langchain_core.messages import AIMessage

from schema import ContactBotState
from llm import generate_response
from database import save_message


def _instruction_greeting() -> str:
    """Instruction for the very first message: greet + ask for name."""
    return (
        "This is the start of a new conversation. "
        "Greet the user warmly on behalf of Incede, briefly explain that you are here "
        "to collect their contact details so the team can reach them, "
        "and ask for their full name."
    )


def ask_name(state: ContactBotState) -> ContactBotState:
    """
    Node that prompts the user for their name.
    On first call: greet + ask for name.
    On retry: LLM explains what went wrong and asks again.
    """
    session_id = state["session_id"]
    is_first_message = not state.get("messages")

    if is_first_message:
        instruction = _instruction_greeting()
    else:
        instruction = "Ask the user for their full name."

    bot_message = generate_response(instruction)
    save_message(session_id, "assistant", bot_message)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=bot_message)],
        "current_field": "name",
        "is_valid": False,
    }
