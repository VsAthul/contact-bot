"""
Node: complete_node
Final node — saves contact details and sends a completion message.
"""

import traceback

from langchain_core.messages import AIMessage

from schema import ContactBotState
from llm import generate_response
from database import save_message, log_error, save_contact_detail, end_session


def _instruction_complete(collected: dict) -> str:
    name = collected.get("name", "the user")
    phone = collected.get("phone", "")
    email = collected.get("email", "")
    description = collected.get("description", "")

    details = f"Name: {name}, Phone: {phone}, Email: {email}"
    if description:
        details += f", Description: {description}"

    return (
        f"All contact details have been successfully collected and saved. "
        f"The details are: {details}. "
        f"Thank {name} warmly by name, confirm these exact details have been recorded, "
        f"and let them know the Incede team will be in touch soon. "
        f"Do not say you are 'assuming' anything — the details are confirmed and saved."
    )


def _instruction_complete_error() -> str:
    """Instruction when saving contact details fails."""
    return (
        "There was a technical error while saving the user's contact details. "
        "Apologise briefly and ask them to try again."
    )


def complete_node(state: ContactBotState) -> ContactBotState:
    """
    Final node that saves collected contact details and sends a completion message.
    The closing message is LLM-generated.
    """
    session_id = state["session_id"]

    try:
        save_contact_detail(
            session_id=session_id,
            name=state.get("name", ""),
            email=state.get("email", ""),
            phone=state.get("phone", ""),
            description=state.get("description"),
        )
        end_session(session_id, contact_collected=True)

        collected = {
            "name": state.get("name"),
            "phone": state.get("phone"),
            "email": state.get("email"),
            "description": state.get("description"),
        }
        bot_message = generate_response(_instruction_complete(collected))
        save_message(session_id, "assistant", bot_message)

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=bot_message)],
            "is_complete": True,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log_error(session_id, type(exc).__name__, str(exc), tb)

        try:
            bot_message = generate_response(_instruction_complete_error())
        except Exception:
            bot_message = "There was an error saving your details. Please try again."

        save_message(session_id, "assistant", bot_message)
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=bot_message)],
            "is_complete": False,
        }
