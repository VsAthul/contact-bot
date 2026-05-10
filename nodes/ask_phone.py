"""
Node: ask_phone
Prompts the user for their phone number.
"""

from langchain_core.messages import AIMessage

from schema import ContactBotState
from llm import generate_response
from database import save_message


def _instruction_ask_phone(collected: dict) -> str:
    name = collected.get("name", "the user")
    return (
        f"The user's name is {name}. "
        "Acknowledge their name warmly and ask for their 10-digit Indian mobile "
        "phone number (starting with 6, 7, 8, or 9, no country code needed)."
    )


def ask_phone(state: ContactBotState) -> ContactBotState:
    """
    Node that prompts the user for their phone number.
    First call: acknowledge name + ask for phone.
    Retry is handled inside llm_node.
    """
    session_id = state["session_id"]
    collected = {
        "name": state.get("name"),
        "phone": state.get("phone"),
        "email": state.get("email"),
    }
    instruction = _instruction_ask_phone(collected)
    bot_message = generate_response(instruction)
    save_message(session_id, "assistant", bot_message)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=bot_message)],
        "current_field": "phone",
        "is_valid": False,
    }
