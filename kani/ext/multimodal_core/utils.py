import fnmatch
import logging
import mimetypes
from typing import IO

import aiohttp

from .exceptions import MediaFormatException

log = logging.getLogger(__name__)


async def get_mime_type(url) -> str:
    """
    Get the MIME type for the content hosted at the given URL.

    First bases it off the URL's file extension, if present; otherwise makes a HEAD request.
    """
    # mimetypes guess
    mime, _ = mimetypes.guess_type(url)
    if mime is not None:
        return mime

    # HEAD request
    async with aiohttp.ClientSession() as session:
        async with session.head(url, allow_redirects=True) as resp:
            return resp.content_type


async def download_media(url: str, f: IO, *, allowed_mime=("image/*", "audio/*", "video/*")):
    """
    Download the content at the given URL to the given file-like object.

    Expects the MIME type of the media to be image/*, audio/*, or video/*. You can override this by passing a list of
    globs in the ``allowed_mime`` parameter (e.g., ``allowed_mime=("*",)`` to allow downloading any media).

    :param url: The URL to download the media from.
    :param f: The file-like object to write the media content to.
    :param allowed_mime: A list of globs that the remote media MIME type must match one of.
    """
    if not allowed_mime:
        raise ValueError("Expected at least one allowed MIME type")
    log.debug(f"Downloading media from url: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            mime = resp.content_type
            if not any(fnmatch.fnmatch(mime, pat) for pat in allowed_mime):
                raise MediaFormatException(f"Invalid MIME type: Expected one of {allowed_mime!r}, got {mime!r}")
            async for chunk in resp.content.iter_chunked(4096):
                f.write(chunk)
