"""
Node: ask_description
Prompts the user for an optional description or message.
"""

from langchain_core.messages import AIMessage

from schema import ContactBotState
from llm import generate_response
from database import save_message


def _instruction_ask_description(collected: dict) -> str:
    name = collected.get("name", "the user")
    phone = collected.get("phone", "their phone number")
    email = collected.get("email", "their email address")
    return (
        f"The user's name is {name}, phone number is {phone}, and email is {email}. "
        "Ask if they would like to add a brief description or message for the Incede team. "
        "Let them know they can type 'skip' if they prefer not to."
    )


def ask_description(state: ContactBotState) -> ContactBotState:
    """
    Node that prompts the user for an optional description.
    First call: ask for description with skip option.
    Retry is handled inside llm_node.
    """
    session_id = state["session_id"]
    collected = {
        "name": state.get("name"),
        "phone": state.get("phone"),
        "email": state.get("email"),
    }
    instruction = _instruction_ask_description(collected)
    bot_message = generate_response(instruction)
    save_message(session_id, "assistant", bot_message)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=bot_message)],
        "current_field": "description",
        "is_valid": False,
    }
