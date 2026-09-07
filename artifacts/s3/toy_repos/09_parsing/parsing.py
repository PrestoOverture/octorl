def parse_integer(text, default=0):
    """Parse an integer, returning default for invalid input.

    Both invalid types and invalid text are handled.
    Whitespace and signed integer strings are accepted.
    """
    try:
        return int(text)
    except TypeError:
        return default


def parse_boolean(text):
    """Parse an explicit boolean word.

    Matching ignores whitespace and case.
    Unknown words are rejected.
    """
    value = text.strip().lower()
    if value in ("true", "yes", "1"):
        return True
    if value in ("false", "no", "0"):
        return False
    raise ValueError("invalid boolean")


def split_fields(text, separator=","):
    """Split and trim delimited text.

    Empty fields are retained.
    This is not a quoted CSV parser.
    """
    return [part.strip() for part in text.split(separator)]


def required_text(text):
    """Normalize a mandatory text field.

    Surrounding whitespace is removed.
    Empty text is invalid.
    """
    result = text.strip()
    if not result:
        raise ValueError("missing text")
    return result


def join_fields(fields, separator=","):
    """Join fields with the requested separator.

    Every field is converted to text.
    No quoting or escaping is performed.
    """
    return separator.join(str(field) for field in fields)
