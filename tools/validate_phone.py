"""
Tool for validating an Indian mobile phone number.
"""

import re
from langchain_core.tools import tool


@tool
def validate_phone(phone: str) -> dict:
    """
    Validate an Indian mobile phone number extracted from user input.

    A valid Indian phone number must:
    - Be exactly 10 digits (after stripping country code +91 or 0 prefix if present)
    - Start with 6, 7, 8, or 9 (valid Indian mobile prefixes)
    - Not be all the same digit (e.g., 9999999999)

    Args:
    phone: The raw phone number string exactly as provided by the user.
           Do NOT extract or truncate digits before passing — pass the full raw input.
    Returns:
        A dictionary with keys:
            - is_valid (bool): Whether the phone number is valid.
            - reason (str or None): Explanation if invalid.
    """
    if not phone or not phone.strip():
        return {"is_valid": False, "reason": "Phone number cannot be empty"}

    # Strip all non-digit characters
    digits_only = re.sub(r"\D", "", phone.strip())

    # Only allow exactly 10 digits — no prefix stripping, no exceptions
    if len(digits_only) != 10:
        return {"is_valid": False, "reason": "Please enter a valid 10-digit Indian mobile number"}

    # Must start with 6, 7, 8, or 9
    if digits_only[0] not in "6789":
        return {"is_valid": False, "reason": "Indian mobile numbers must start with 6, 7, 8, or 9"}

    # Reject all identical digits (e.g., 9999999999)
    if len(set(digits_only)) == 1:
        return {"is_valid": False, "reason": "Please enter a real phone number"}

    return {"is_valid": True, "reason": None}
