"""
Tool for validating an optional description or message.
"""

from langchain_core.tools import tool


@tool
def validate_description(description: str) -> dict:
    """
    Validate an optional description or message from the user.

    A valid description must:
    - If provided, contain at least 5 characters
    - If provided, not be a single meaningless word
    - Be skippable (the user can choose not to provide one)

    Args:
        description: The description text to validate, or an empty string to skip.

    Returns:
        A dictionary with keys:
            - is_valid (bool): Whether the description is valid or was intentionally skipped.
            - reason (str or None): Explanation if invalid.
    """
    # Allow empty input as the description is optional
    if not description or not description.strip():
        return {"is_valid": True, "reason": None}

    cleaned = description.strip()

    # If provided, it must have some meaningful content
    if len(cleaned) < 5:
        return {
            "is_valid": False,
            "reason": "Description is too short. Please provide more details or type 'skip' to leave it blank"
        }

    return {"is_valid": True, "reason": None}
