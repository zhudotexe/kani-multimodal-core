from kani.exceptions import KaniException


class MediaFormatException(KaniException):
    """Encountered an invalid MIME type dowloading or processing multimodal media."""
