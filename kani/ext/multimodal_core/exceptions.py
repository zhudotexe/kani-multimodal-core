from kani.exceptions import KaniException

__all__ = ("MediaFormatException",)


class MediaFormatException(KaniException):
    """Encountered an invalid MIME type dowloading or processing multimodal media."""
