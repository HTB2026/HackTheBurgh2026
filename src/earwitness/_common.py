from enum import StrEnum
from typing import TypeIs


class ForensicError(StrEnum):
    """Base for every domain error enum. Subclasses add the members."""


# NOTE: This alias cannot live inside ForensicError, because an Enum body turns
# every non-descriptor attribute (including a type alias) into a member.
type OnSuccess[T] = T | ForensicError


def is_error[T](value: T | ForensicError) -> TypeIs[ForensicError]:
    return isinstance(value, ForensicError)


def is_success[T](value: T | ForensicError) -> TypeIs[T]:
    return not isinstance(value, ForensicError)
