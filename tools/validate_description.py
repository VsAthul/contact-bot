from langchain_core.tools import tool


@tool
def validate_description(description: str) -> dict:
    """
    Validate an optional description/message.

    Returns:
        {
            "is_valid": bool,
            "reason": str | None,
        }

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

    try:
        if description is None:
            return {
                "is_valid": True,
                "reason": None,
            }

        text = str(description).strip()

        # User intentionally skipped
        skip_values = {
            "skip",
            "none",
            "n/a",
            "na",
            "no",
            "nothing",
            "skip it",
            "",
        }

        if text.lower() in skip_values:
            return {
                "is_valid": True,
                "reason": None,
            }

        # Too short
        if len(text) < 5:
            return {
                "is_valid": False,
                "reason": (
                    "Please provide a slightly more detailed description "
                    "or type 'skip'."
                ),
            }

        return {
            "is_valid": True,
            "reason": None,
        }

    except Exception as exc:
        return {
            "is_valid": False,
            "reason": str(exc),
        }