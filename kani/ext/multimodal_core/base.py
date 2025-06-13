import functools

from kani import MessagePart
from pydantic import ConfigDict


class BaseMultimodalPart(MessagePart):
    model_config = ConfigDict(ignored_types=(functools.cached_property,))

    extra: dict = {}
    """
    Specific engines may store additional extra data in this dictionary. See an engine's documentation for details about
    any extras it may store or expect.
    """


class TextPart(BaseMultimodalPart):
    """A part representing basic text data."""

    text: str

    def __str__(self):
        return self.text
