"""
Node: ask_email
Prompts the user for their email address.
"""

from langchain_core.messages import AIMessage

from schema import ContactBotState
from llm import generate_response
from database import save_message


def _instruction_ask_email(collected: dict) -> str:
    name = collected.get("name", "the user")
    phone = collected.get("phone", "their phone number")
    return (
        f"The user's name is {name} and their phone number is {phone}. "
        "Acknowledge that you have their phone number and ask for their email address."
    )


def ask_email(state: ContactBotState) -> ContactBotState:
    """
    Node that prompts the user for their email address.
    First call: acknowledge phone + ask for email.
    Retry is handled inside llm_node.
    """
    session_id = state["session_id"]
    collected = {
        "name": state.get("name"),
        "phone": state.get("phone"),
        "email": state.get("email"),
    }
    instruction = _instruction_ask_email(collected)
    bot_message = generate_response(instruction)
    save_message(session_id, "assistant", bot_message)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=bot_message)],
        "current_field": "email",
        "is_valid": False,
    }
