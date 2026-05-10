"""
Tools package for the Incede contact bot.
Exports all validation tools and the combined list for tool calling.
"""

from tools.validate_name import validate_name
from tools.validate_phone import validate_phone
from tools.validate_email import validate_email
from tools.validate_description import validate_description

ALL_VALIDATION_TOOLS = [validate_name, validate_phone, validate_email, validate_description]

__all__ = [
    "validate_name",
    "validate_phone",
    "validate_email",
    "validate_description",
    "ALL_VALIDATION_TOOLS",
]
