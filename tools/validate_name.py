"""
Tool for validating a person's name.
"""

import re
from langchain_core.tools import tool


@tool
def validate_name(name: str) -> dict:
    """
    Validate a person's name extracted from user input.

    A valid name must:
    - Contain at least 2 characters
    - Contain only alphabetic characters, spaces, hyphens, or apostrophes
    - Not be a generic placeholder like 'name', 'user', or 'na'
    - Have at least one actual word with 2 or more letters

    Args:
        name: The name string to validate.

    Returns:
        A dictionary with keys:
            - is_valid (bool): Whether the name is valid.
            - reason (str or None): Explanation if invalid.
    """
    if not name or not name.strip():
        return {"is_valid": False, "reason": "Name cannot be empty"}

    cleaned = name.strip()

    # Check minimum length
    if len(cleaned) < 2:
        return {"is_valid": False, "reason": "Name is too short, please provide your full name"}

    # Check for valid characters only
    if not re.match(r"^[A-Za-z\s\-']+$", cleaned):
        return {"is_valid": False, "reason": "Name should only contain letters, spaces, hyphens, or apostrophes"}

    # Check for placeholder or nonsense names
    invalid_names = {"name", "user", "na", "n/a", "none", "test", "unknown", "anonymous"}
    if cleaned.lower() in invalid_names:
        return {"is_valid": False, "reason": "Please provide your actual name"}

    # Ensure at least one word has 2 or more characters
    words = [w for w in cleaned.split() if len(w) >= 2]
    if not words:
        return {"is_valid": False, "reason": "Please provide a valid name with at least one full word"}

    return {"is_valid": True, "reason": None}
