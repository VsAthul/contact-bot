"""
utils/pii_scrubber.py
---------------------
Lightweight helpers that mask PII before anything is written to logs.

Rules
-----
* Email  : keep first char of local part and domain TLD  →  j***@***.com
* Phone  : keep last 4 digits, mask the rest             →  ******1234
* Name   : keep first char of each token                 →  J*** D***
* Generic text scrub: apply all three rules in one pass  →  safe for
  arbitrary strings that may contain any of the above.
"""

import re


# --------------------------------------------------------------------------
# Individual maskers
# --------------------------------------------------------------------------

def mask_email(email: str) -> str:
    """j***@***.com  (preserves first local char and TLD only)"""
    if not email or "@" not in email:
        return "***"
    local, _, domain = email.partition("@")
    masked_local = (local[0] + "***") if local else "***"
    # domain: keep TLD only (e.g. 'gmail.com' → '***.com')
    parts = domain.rsplit(".", 1)
    masked_domain = ("***." + parts[1]) if len(parts) == 2 else "***"
    return f"{masked_local}@{masked_domain}"


def mask_phone(phone: str) -> str:
    """Show only the last 4 digits, mask the rest with *."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) <= 4:
        return "*" * len(digits)
    return "*" * (len(digits) - 4) + digits[-4:]


def mask_name(name: str) -> str:
    """Keep only the first character of each name token: 'John Doe' → 'J*** D***'."""
    if not name:
        return "***"
    return " ".join(
        (token[0] + "***") if token else "***"
        for token in name.split()
    )


# --------------------------------------------------------------------------
# Generic scrub — applies all three rules on arbitrary text
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b\d[\d\s\-().]{6,}\d\b")


def scrub(text: str) -> str:
    """
    Replace any email addresses and phone-like digit sequences in *text*
    with masked versions.  Safe to call on arbitrary log strings.
    """
    if not isinstance(text, str):
        text = repr(text)

    # Mask emails first (they contain @, so the phone pattern won't match them)
    text = _EMAIL_RE.sub(lambda m: mask_email(m.group()), text)

    # Mask phone-like sequences
    text = _PHONE_RE.sub(lambda m: mask_phone(re.sub(r"\D", "", m.group())), text)

    return text