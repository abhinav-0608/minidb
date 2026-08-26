"""Exception types shared across MiniDB."""


class MiniDBError(Exception):
    """Base class for every error MiniDB raises on purpose.

    One exception type on purpose: the messages carry the detail. Callers
    that want to distinguish cases can match on the message or on where the
    call was made, not on a subclass hierarchy.
    """
