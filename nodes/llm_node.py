"""
Node: llm_node
Central LLM processing node — handles extraction, validation, and reply generation.
"""

import re
import traceback

from flask import current_app
from langchain_core.messages import AIMessage, HumanMessage

from schema import (
    ContactBotState,
    ExtractedName,
    ExtractedPhone,
    ExtractedEmail,
    ExtractedDescription,
)
from llm import get_structured_llm, get_validation_llm, generate_response
from tools import (
    validate_name,
    validate_phone,
    validate_email,
    validate_description,
    ALL_VALIDATION_TOOLS,
)
from database import save_message, log_error
from utils.pii_scrubber import mask_email, mask_phone, mask_name, scrub


# --------------------------------------------------------------------------
# Helper: run validation via LLM tool calling
# --------------------------------------------------------------------------

def _run_validation_via_llm(field: str, value: str) -> dict:
    """
    Ask the LLM to validate a value by calling the appropriate tool.

    The LLM is bound to all four validation tools and is prompted to call
    the one that matches the current field.  We then execute whichever
    tool the LLM chose and return its result dict.

    Args:
        field: One of 'name', 'phone', 'email', or 'description'.
        value: The extracted value to validate.

    Returns:
        A dictionary with is_valid (bool) and reason (str or None).
    """
    tool_map = {
        "validate_name":        validate_name,
        "validate_phone":       validate_phone,
        "validate_email":       validate_email,
        "validate_description": validate_description,
    }

    prompt = (
        f"You must validate the user's {field} using the correct validation tool. "
        f"The {field} to validate is: '{value}'. "
        f"Call the appropriate tool now with this value."
    )

    validation_llm = get_validation_llm(ALL_VALIDATION_TOOLS)
    response = validation_llm.invoke([HumanMessage(content=prompt)])

    # Log tool_calls count only — the raw payload may contain PII
    num_calls = len(response.tool_calls) if response.tool_calls else 0
    current_app.logger.debug(
        "validation tool_calls count=%d field='%s'", num_calls, field
    )

    if not response.tool_calls:
        current_app.logger.warning(
            "LLM did not call a tool for field='%s'", field
        )
        return {
            "is_valid": False,
            "reason": "Validation could not be performed — please try again.",
        }

    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # Scrub any PII that might appear in tool_args before logging
    current_app.logger.debug(
        "LLM chose tool='%s' args='%s'", tool_name, scrub(str(tool_args))
    )

    tool_fn = tool_map.get(tool_name)
    if not tool_fn:
        current_app.logger.warning("LLM called unknown tool '%s'", tool_name)
        return {
            "is_valid": False,
            "reason": f"Unknown validation tool called: {tool_name}",
        }

    result = tool_fn.invoke(tool_args)
    return result


# --------------------------------------------------------------------------
# Instruction builders
# --------------------------------------------------------------------------

def _instruction_invalid(field: str, value: str, reason: str, collected: dict) -> str:
    name = collected.get("name", "the user")
    field_labels = {
        "name": "full name",
        "phone": "10-digit Indian mobile phone number (starting with 6, 7, 8, or 9)",
        "email": "email address",
        "description": "description or message",
    }
    label = field_labels.get(field, field)
    return (
        f"The user submitted '{value}' as their {field} but it failed validation. "
        f"Reason: {reason}. "
        f"Their name is {name}. "
        f"Politely let them know that input wasn't valid, briefly explain why "
        f"({reason}), and ask them to provide a valid {label}."
    )


def _instruction_extraction_failed(field: str, raw_input: str, collected: dict) -> str:
    name = collected.get("name", "the user")
    field_labels = {
        "name": "full name",
        "phone": "10-digit Indian mobile phone number",
        "email": "email address",
        "description": "description",
    }
    label = field_labels.get(field, field)
    return (
        f"The user typed '{raw_input}' but no {label} could be found in it. "
        f"Their name is {name}. "
        f"Gently tell them you could not find a {label} in what they typed "
        f"and ask them to provide their {label} clearly."
    )


def _instruction_off_topic(field: str, raw_input: str, collected: dict) -> str:
    name = collected.get("name", "the user")
    field_labels = {
        "name": "full name",
        "phone": "phone number",
        "email": "email address",
        "description": "description",
    }
    label = field_labels.get(field, field)
    return (
        f"The user ({name}) sent an off-topic message: '{raw_input}'. "
        f"Politely explain that you can only help with collecting contact details, "
        f"and redirect them by asking for their {label}."
    )


# --------------------------------------------------------------------------
# PII-safe log helpers
# --------------------------------------------------------------------------

def _log_extracted(field: str, raw_input: str, extracted) -> None:
    """Log extraction result with PII masked based on field type."""
    safe_raw = scrub(raw_input)

    if field == "email":
        safe_value = mask_email(extracted.email) if (extracted.found and extracted.email) else "null"
    elif field == "phone":
        safe_value = mask_phone(extracted.phone) if (extracted.found and extracted.phone) else "null"
    elif field == "name":
        safe_value = mask_name(extracted.name) if (extracted.found and extracted.name) else "null"
    else:
        # description — not PII in the same sense, but still scrub it
        desc = getattr(extracted, "description", None)
        safe_value = scrub(desc) if desc else ("skipped" if getattr(extracted, "skipped", False) else "null")

    current_app.logger.debug(
        "extraction field='%s' raw_input='%s' extracted_value='%s'",
        field, safe_raw, safe_value,
    )


def _log_extracted_value(field: str, value) -> None:
    """Log the final extracted_value with PII masked."""
    if field == "email" and value:
        safe = mask_email(value)
    elif field == "phone" and value:
        safe = mask_phone(value)
    elif field == "name" and value:
        safe = mask_name(value)
    else:
        safe = scrub(str(value)) if value else repr(value)

    length = len(value) if value else 0
    current_app.logger.debug(
        "extracted_value field='%s' value='%s' len=%d", field, safe, length
    )


# --------------------------------------------------------------------------
# Node function
# --------------------------------------------------------------------------

def llm_node(state: ContactBotState) -> ContactBotState:
    """
    Central LLM processing node that:
    1. Detects and handles off-topic input.
    2. Extracts the relevant value from user input using structured output.
    3. Validates the extracted value by having the LLM call the appropriate tool.
    4. Generates a contextual LLM reply for every outcome.
    """
    session_id = state["session_id"]
    current_field = state.get("current_field", "name")
    raw_input = state.get("raw_user_input", "")

    collected = {
        "name": state.get("name"),
        "phone": state.get("phone"),
        "email": state.get("email"),
        "description": state.get("description"),
    }

    try:
        # ------------------------------------------------------------------
        # Step 1: Off-topic detection
        # ------------------------------------------------------------------
        def _is_clearly_off_topic(text: str, field: str) -> bool:
            stripped = text.strip()
            if not stripped.endswith("?"):
                return False
            if field == "name" and re.match(r"^[a-zA-Z\s\-']+\?$", stripped):
                return False
            if re.search(r"\d", stripped):
                return False
            if "@" in stripped:
                return False
            if len(stripped) > 60:
                return False
            return True

        if _is_clearly_off_topic(raw_input, current_field):
            instruction = _instruction_off_topic(current_field, raw_input, collected)
            bot_message = generate_response(instruction)
            save_message(session_id, "assistant", bot_message)
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=bot_message)],
                "is_valid": False,
                "error_message": None,
            }

        # ------------------------------------------------------------------
        # Step 2: Extract the value using structured output
        # ------------------------------------------------------------------
        extraction_schema_map = {
            "name": ExtractedName,
            "phone": ExtractedPhone,
            "email": ExtractedEmail,
            "description": ExtractedDescription,
        }

        extraction_prompt_map = {
            "name": (
                f"Extract the person's name from this input and return a json object with exactly these fields: "
                f"'name' (the extracted name as a string) and 'found' (boolean true or false). "
                f"A name can be a single first name (e.g. 'athul'), full name, or embedded in a sentence. "
                f"If a name is present, set found=true and name='<the name>'. "
                f"If no name is present, set found=false and name=null. "
                f"Input: '{raw_input}'"
            ),
            "phone": (
                f"Extract the phone number from this input and return a json object with exactly these fields: "
                f"'phone' (the extracted number as a string) and 'found' (boolean true or false). "
                f"Copy the digits exactly as typed into the 'phone' field — do NOT validate or modify them. "
                f"If a phone number is present, set found=true and phone='<the number>'. "
                f"If no phone number is present, set found=false and phone=null. "
                f"Input: '{raw_input}'"
            ),
            "email": (
                f"Extract the email address from this input and return a json object with exactly these fields: "
                f"'email' (the extracted email as a string) and 'found' (boolean true or false). "
                f"If an email is present, set found=true and email='<the email>'. "
                f"If no email is present, set found=false and email=null. "
                f"Input: '{raw_input}'"
            ),
            "description": (
                f"Extract the description/message from this input and return a json object with exactly these fields: "
                f"'description' (the extracted text as a string) and 'skipped' (boolean true or false). "
                f"The user wants to skip if they typed any of: 'skip', 'no', 'none', 'nothing', 'skip it', 'no thanks', or anything that means they don't want to add a description. "
                f"If the user wants to skip, set skipped=true and description=null. "
                f"Otherwise set skipped=false and description='<their message>'. "
                f"Input: '{raw_input}'"
            ),
        }

        schema = extraction_schema_map[current_field]
        prompt = extraction_prompt_map[current_field]

        structured_llm = get_structured_llm(schema)
        extracted = structured_llm.invoke(prompt)

        # Log extraction result with PII masked
        _log_extracted(current_field, raw_input, extracted)

        # ------------------------------------------------------------------
        # Step 3: Get the extracted value
        # ------------------------------------------------------------------
        extracted_value = None

        if current_field == "name":
            extracted_value = extracted.name if extracted.found else None
        elif current_field == "phone":
            extracted_value = extracted.phone if extracted.found else None
        elif current_field == "email":
            extracted_value = extracted.email if extracted.found else None
        elif current_field == "description":
            if extracted.skipped:
                extracted_value = ""
            elif extracted.description:
                extracted_value = extracted.description
            else:
                extracted_value = ""  # fallback — treat as skip

        # Nothing extractable found
        if extracted_value is None and current_field != "description":
            instruction = _instruction_extraction_failed(current_field, raw_input, collected)
            bot_message = generate_response(instruction)
            save_message(session_id, "assistant", bot_message)
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=bot_message)],
                "is_valid": False,
                "error_message": f"Could not extract {current_field} from user input",
            }

        # ------------------------------------------------------------------
        # Step 4: Validate via LLM tool calling
        # ------------------------------------------------------------------
        _log_extracted_value(current_field, extracted_value)
        validation_result = _run_validation_via_llm(current_field, extracted_value or "")
        is_valid = validation_result.get("is_valid", False)
        reason = validation_result.get("reason") or "The value did not pass validation."

        # ------------------------------------------------------------------
        # Step 5: Generate a contextual LLM reply and update state
        # ------------------------------------------------------------------
        updated_state = dict(state)
        updated_state["is_valid"] = is_valid
        updated_state["error_message"] = reason if not is_valid else None

        if is_valid:
            if current_field == "name":
                updated_state["name"] = extracted_value
                collected["name"] = extracted_value
            elif current_field == "phone":
                updated_state["phone"] = extracted_value
                collected["phone"] = extracted_value
            elif current_field == "email":
                updated_state["email"] = extracted_value
                collected["email"] = extracted_value
            elif current_field == "description":
                updated_state["description"] = extracted_value
                collected["description"] = extracted_value
        else:
            instruction = _instruction_invalid(
                current_field, extracted_value or raw_input, reason, collected
            )
            bot_message = generate_response(instruction)
            save_message(session_id, "assistant", bot_message)
            updated_state["messages"] = state["messages"] + [AIMessage(content=bot_message)]

        return updated_state

    except Exception as exc:
        tb = traceback.format_exc()
        log_error(session_id, type(exc).__name__, str(exc), tb)

        try:
            bot_message = generate_response(
                f"An unexpected technical error occurred: {exc}. "
                f"Apologise briefly and ask the user to try again."
            )
        except Exception:
            bot_message = "I ran into a technical issue. Could you please try again?"

        save_message(session_id, "assistant", bot_message)
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=bot_message)],
            "is_valid": False,
            "error_message": str(exc),
        }