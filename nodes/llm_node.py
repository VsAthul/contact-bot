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
    the one that matches the current field. We then execute whichever
    tool the LLM chose and return its result dict.
    """

    tool_map = {
        "validate_name": validate_name,
        "validate_phone": validate_phone,
        "validate_email": validate_email,
        "validate_description": validate_description,
    }

    prompt = (
        f"You must validate the user's {field} using the correct validation tool. "
        f"The {field} to validate is: '{value}'. "
        f"Call the appropriate tool now with this value."
    )

    validation_llm = get_validation_llm(ALL_VALIDATION_TOOLS)

    try:
        response = validation_llm.invoke([HumanMessage(content=prompt)])

    except Exception as exc:
        current_app.logger.exception(
            "validation llm invocation failed for field='%s': %s",
            field,
            exc,
        )

        return {
            "is_valid": False,
            "reason": "Validation could not be performed — please try again.",
        }

    # Log tool_calls count only — raw payload may contain PII
    num_calls = len(response.tool_calls) if response.tool_calls else 0

    current_app.logger.debug(
        "validation tool_calls count=%d field='%s'",
        num_calls,
        field,
    )

    if not response.tool_calls:
        current_app.logger.warning(
            "LLM did not call a validation tool for field='%s'",
            field,
        )

        return {
            "is_valid": False,
            "reason": "Validation could not be performed — please try again.",
        }

    tool_call = response.tool_calls[0]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})

    current_app.logger.debug(
        "LLM chose tool='%s' args='%s'",
        tool_name,
        scrub(str(tool_args)),
    )

    tool_fn = tool_map.get(tool_name)

    if not tool_fn:
        current_app.logger.warning(
            "LLM called unknown validation tool '%s'",
            tool_name,
        )

        return {
            "is_valid": False,
            "reason": f"Unknown validation tool called: {tool_name}",
        }

    try:
        result = tool_fn.invoke(tool_args)

    except Exception as exc:
        current_app.logger.exception(
            "validation tool execution failed tool='%s': %s",
            tool_name,
            exc,
        )

        return {
            "is_valid": False,
            "reason": "Validation failed due to an internal error.",
        }

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


def _instruction_extraction_failed(
    field: str,
    raw_input: str,
    collected: dict,
) -> str:
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
        f"The user ({name}) sent a message that did not clearly provide their "
        f"requested {label}: '{raw_input}'. "
        f"Politely explain that you are collecting contact details and ask them "
        f"to provide their {label} clearly."
    )


# --------------------------------------------------------------------------
# PII-safe log helpers
# --------------------------------------------------------------------------

def _log_extracted(field: str, raw_input: str, extracted) -> None:
    """Log extraction result with PII masked based on field type."""

    safe_raw = scrub(raw_input)

    if field == "email":
        safe_value = (
            mask_email(extracted.email)
            if (getattr(extracted, "found", False) and extracted.email)
            else "null"
        )

    elif field == "phone":
        safe_value = (
            mask_phone(extracted.phone)
            if (getattr(extracted, "found", False) and extracted.phone)
            else "null"
        )

    elif field == "name":
        safe_value = (
            mask_name(extracted.name)
            if (getattr(extracted, "found", False) and extracted.name)
            else "null"
        )

    else:
        desc = getattr(extracted, "description", None)

        safe_value = (
            scrub(desc)
            if desc
            else (
                "skipped"
                if getattr(extracted, "skipped", False)
                else "null"
            )
        )

    current_app.logger.debug(
        "extraction field='%s' raw_input='%s' extracted_value='%s'",
        field,
        safe_raw,
        safe_value,
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
        "extracted_value field='%s' value='%s' len=%d",
        field,
        safe,
        length,
    )


# --------------------------------------------------------------------------
# Node function
# --------------------------------------------------------------------------

def llm_node(state: ContactBotState) -> ContactBotState:
    """
    Central LLM processing node that:
    1. Performs lightweight unusable-input filtering.
    2. Extracts the relevant value using structured output.
    3. Uses extraction intent classification instead of regex heuristics.
    4. Validates extracted values through LLM tool calling.
    5. Generates contextual responses.
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
        # Step 1: Minimal unusable-input detection
        # ------------------------------------------------------------------

        def _is_clearly_off_topic(text: str) -> bool:
            """
            Extremely conservative fast-path filter.

            The structured extraction layer is responsible for determining
            whether the user is answering the prompt, asking a question,
            sending an unclear message, or going off-topic.

            This helper only rejects inputs that are obviously unusable.
            """

            try:
                if text is None:
                    return True

                stripped = text.strip()

                # Empty / whitespace-only input
                if not stripped:
                    return True

                # Single emoji / symbol-only payloads
                if len(stripped) <= 3:
                    symbol_only = re.fullmatch(
                        r"[^\w\s]+",
                        stripped,
                        flags=re.UNICODE,
                    )

                    if symbol_only:
                        return True

                return False

            except Exception as exc:
                current_app.logger.exception(
                    "off-topic heuristic failure: %s",
                    exc,
                )

                # Fail open
                return False

        if _is_clearly_off_topic(raw_input):

            instruction = _instruction_off_topic(
                current_field,
                raw_input,
                collected,
            )

            bot_message = generate_response(instruction)

            save_message(session_id, "assistant", bot_message)

            return {
                **state,
                "messages": state["messages"] + [
                    AIMessage(content=bot_message)
                ],
                "is_valid": False,
                "error_message": None,
            }

        # ------------------------------------------------------------------
        # Step 2: Structured extraction
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
                f"'name' (string or null), "
                f"'found' (boolean), and "
                f"'intent' (one of: 'answer', 'question', 'off_topic', 'unclear'). "
                f"A name may be a first name, full name, or embedded inside a sentence. "
                f"If the user is asking what information is needed or is not attempting to provide a name, "
                f"set found=false and select the appropriate intent. "
                f"Input: '{raw_input}'"
            ),

            "phone": (
                f"Extract the phone number from this input and return a json object with exactly these fields: "
                f"'phone' (string or null), "
                f"'found' (boolean), and "
                f"'intent' (one of: 'answer', 'question', 'off_topic', 'unclear'). "
                f"Copy the phone digits exactly as written without validating or modifying them. "
                f"If the user is not attempting to provide a phone number, set found=false. "
                f"Input: '{raw_input}'"
            ),

            "email": (
                f"Extract the email address from this input and return a json object with exactly these fields: "
                f"'email' (string or null), "
                f"'found' (boolean), and "
                f"'intent' (one of: 'answer', 'question', 'off_topic', 'unclear'). "
                f"If the user is not attempting to provide an email address, set found=false. "
                f"Input: '{raw_input}'"
            ),

            "description": (
                f"Extract the description/message from this input and return a json object with exactly these fields: "
                f"'description' (string or null), "
                f"'skipped' (boolean), "
                f"'found' (boolean), and "
                f"'intent' (one of: 'answer', 'question', 'off_topic', 'unclear'). "
                f"The user wants to skip if they typed any of: 'skip', 'no', 'none', 'nothing', 'skip it', or similar wording. "
                f"If the user wants to skip, set skipped=true and description=null. "
                f"If the user asks a question instead of providing a description, set found=false and choose the proper intent. "
                f"Otherwise extract the user's message into description. "
                f"Input: '{raw_input}'"
            ),
        }

        schema = extraction_schema_map[current_field]
        prompt = extraction_prompt_map[current_field]

        try:
            structured_llm = get_structured_llm(schema)

            extracted = structured_llm.invoke(
                [HumanMessage(content=prompt)]
            )

        except Exception as exc:

            current_app.logger.exception(
                "structured extraction failed for field='%s': %s",
                current_field,
                exc,
            )

            log_error(
                session_id,
                "Structured extraction failure",
                str(exc),
                traceback.format_exc(),
            )

            fallback_message = (
                "I had trouble understanding your response. "
                "Could you please try again?"
            )

            save_message(session_id, "assistant", fallback_message)

            return {
                **state,
                "messages": state["messages"] + [
                    AIMessage(content=fallback_message)
                ],
                "is_valid": False,
                "error_message": "Structured extraction failure",
            }

        # Log extraction safely
        _log_extracted(current_field, raw_input, extracted)

        # ------------------------------------------------------------------
        # Step 3: Intent handling
        # ------------------------------------------------------------------

        extraction_intent = getattr(extracted, "intent", "answer")

        if extraction_intent in {"question", "off_topic", "unclear"}:

            instruction = _instruction_off_topic(
                current_field,
                raw_input,
                collected,
            )

            bot_message = generate_response(instruction)

            save_message(session_id, "assistant", bot_message)

            return {
                **state,
                "messages": state["messages"] + [
                    AIMessage(content=bot_message)
                ],
                "is_valid": False,
                "error_message": None,
            }

        # ------------------------------------------------------------------
        # Step 4: Extract actual value
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
                extracted_value = ""

        # ------------------------------------------------------------------
        # Step 5: Extraction failure handling
        # ------------------------------------------------------------------

        if extracted_value is None and current_field != "description":

            instruction = _instruction_extraction_failed(
                current_field,
                raw_input,
                collected,
            )

            bot_message = generate_response(instruction)

            save_message(session_id, "assistant", bot_message)

            return {
                **state,
                "messages": state["messages"] + [
                    AIMessage(content=bot_message)
                ],
                "is_valid": False,
                "error_message": (
                    f"Could not extract {current_field} from user input"
                ),
            }

        # ------------------------------------------------------------------
        # Step 6: Validation
        # ------------------------------------------------------------------

        _log_extracted_value(current_field, extracted_value)

        validation_result = _run_validation_via_llm(
            current_field,
            extracted_value or "",
        )

        is_valid = validation_result.get("is_valid", False)

        reason = (
            validation_result.get("reason")
            or "The value did not pass validation."
        )

        # ------------------------------------------------------------------
        # Step 7: Update state and respond
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
                current_field,
                extracted_value or raw_input,
                reason,
                collected,
            )

            bot_message = generate_response(instruction)

            save_message(session_id, "assistant", bot_message)

            updated_state["messages"] = state["messages"] + [
                AIMessage(content=bot_message)
            ]

        return updated_state

    except Exception as exc:

        tb = traceback.format_exc()

        log_error(
            session_id,
            type(exc).__name__,
            str(exc),
            tb,
        )

        current_app.logger.exception(
            "unexpected llm_node failure: %s",
            exc,
        )

        try:
            bot_message = generate_response(
                f"An unexpected technical error occurred: {exc}. "
                f"Apologise briefly and ask the user to try again."
            )

        except Exception:
            bot_message = (
                "I ran into a technical issue. "
                "Could you please try again?"
            )

        save_message(session_id, "assistant", bot_message)

        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=bot_message)
            ],
            "is_valid": False,
            "error_message": str(exc),
        }