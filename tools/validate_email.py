"""
Tool for validating an email address.
"""

import re
from langchain_core.tools import tool


@tool
def validate_email(email: str) -> dict:
    """
    Validate an email address extracted from user input.

    A valid email address must:
    - Follow the standard format: localpart@domain.tld
    - Have a local part of at least 2 characters
    - Have a domain with a known, real-looking TLD (2–6 chars)
    - Have a domain label of at least 3 characters (e.g. gmail, yahoo)
    - Contain no spaces anywhere
    - Not use a single-character or suspicious local part

    Args:
        email: The email address string to validate.

    Returns:
        A dictionary with keys:
            - is_valid (bool): Whether the email is valid.
            - reason (str or None): Explanation if invalid.
    """
    if not email or not email.strip():
        return {"is_valid": False, "reason": "Email address cannot be empty"}

    # Reject immediately if the raw input contains spaces — catches "athuk @ gmail . com"
    if " " in email:
        return {"is_valid": False, "reason": "Email address must not contain spaces"}

    cleaned = email.strip().lower()

    # Must have exactly one @
    if cleaned.count("@") != 1:
        return {"is_valid": False, "reason": "Email address must contain exactly one @ symbol"}

    local, domain = cleaned.split("@")

    # Local part validations
    if not local:
        return {"is_valid": False, "reason": "Email address is missing the part before @"}

    if len(local) < 2:
        return {"is_valid": False, "reason": "Email local part is too short (e.g. use john@example.com)"}

    if not re.match(r"^[a-z0-9][a-z0-9._%+\-]*[a-z0-9]$", local) and len(local) > 1:
        return {"is_valid": False, "reason": "Email local part contains invalid characters or formatting"}

    # Dots cannot be consecutive or at the start/end of the local part
    if ".." in local or local.startswith(".") or local.endswith("."):
        return {"is_valid": False, "reason": "Email local part has invalid dot placement"}

    # Domain validations
    if not domain:
        return {"is_valid": False, "reason": "Email address is missing the domain part"}

    if "." not in domain:
        return {"is_valid": False, "reason": "Email domain must contain a dot (e.g. gmail.com)"}

    if domain.startswith(".") or domain.endswith("."):
        return {"is_valid": False, "reason": "Email domain has invalid dot placement"}

    if ".." in domain:
        return {"is_valid": False, "reason": "Email domain has consecutive dots"}

    domain_parts = domain.split(".")
    tld = domain_parts[-1]
    domain_label = domain_parts[-2] if len(domain_parts) >= 2 else ""

    # TLD must be 2–6 alphabetic characters (e.g. com, in, org, co.in)
    if not re.match(r"^[a-z]{2,6}$", tld):
        return {"is_valid": False, "reason": "Email has an invalid or unrecognised top-level domain"}

    # Domain label (e.g. "gmail" in gmail.com) must be at least 3 characters
    if len(domain_label) < 3:
        return {"is_valid": False, "reason": "Email domain name is too short to be valid (e.g. use gmail.com)"}

    # Domain label must be alphanumeric (hyphens allowed inside, not at edges)
    if not re.match(r"^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$", domain_label):
        return {"is_valid": False, "reason": "Email domain contains invalid characters"}

    return {"is_valid": True, "reason": None}
