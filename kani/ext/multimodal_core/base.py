import abc
import base64
import functools
import hashlib
import io
import mimetypes
import os
import re
import tempfile
import zlib

from kani import MessagePart
from kani.utils import saveload
from kani.utils.typing import PathLike
from pydantic import ConfigDict, model_serializer, model_validator

from .types import BinaryFileLike
from .utils import download_media


# ==== bases ====
class BaseMultimodalPart(MessagePart, abc.ABC):
    model_config = ConfigDict(ignored_types=(functools.cached_property,))


class BinaryFilePart(BaseMultimodalPart, arbitrary_types_allowed=True):
    """
    A MessagePart containing arbitrary binary data.

    The raw data is saved as a file-like object and a MIME type. This allows applications to persist large files on
    disk (using a FileIO) or in memory (using a BytesIO).

    When serialized, the binary is represented as a data URI. This can lead to some really big files!
    """

    file: BinaryFileLike
    """The readable binary file-like object containing the data."""

    mime: str
    """The MIME file type of the file."""

    # ---- hashing helpers ----
    # the file's mtime the last time we hashed it, used to see if we need to rehash
    _last_hash_mtime: float = None
    _last_hash: bytes = None

    # ==== constructors ====
    @classmethod
    def from_file(cls, fp: PathLike | BinaryFileLike, mime: str = None, **kwargs):
        """
        Create a BinaryFilePart from a local file.

        :param fp: The path to the file, or a file-like object.
        :param mime: The MIME file type (https://www.iana.org/assignments/media-types/media-types.xhtml)
            of the file. If not passed, will attempt to guess the filetype from the file name.
        """
        # file-like object
        if isinstance(fp, BinaryFileLike):
            if hasattr(fp, "name") and isinstance(fp.name, str):  # fp.name can be int in the case of unnamed temp files
                mime, encoding = mimetypes.guess_type(fp.name)
            if mime is None:
                raise ValueError(
                    "The filetype cannot be guessed from the data when passing a file-like object to"
                    " BinaryFilePart.from_file. Please pass the `mime` parameter with an IANA-defined media"
                    " type (https://www.iana.org/assignments/media-types/media-types.xhtml)."
                )
            return cls(file=fp, mime=mime, **kwargs)

        if mime is None:
            mime, encoding = mimetypes.guess_type(fp)
            if mime is None:
                raise ValueError(
                    f"The file type of {fp!r} could not be determined. Please pass the `mime` parameter with an"
                    " IANA-defined media type (https://www.iana.org/assignments/media-types/media-types.xhtml)."
                )

        handle = open(fp, mode="rb")
        return cls(file=handle, mime=mime, **kwargs)

    @classmethod
    def from_bytes(cls, data: bytes, mime: str, **kwargs):
        """
        Create a BinaryFilePart from raw bytes.

        :param data: The bytes.
        :param mime: The MIME file type (https://www.iana.org/assignments/media-types/media-types.xhtml)
            of the file.
        """
        handle = io.BytesIO(data)
        return cls(file=handle, mime=mime, **kwargs)

    @classmethod
    def from_b64(cls, data: str, mime: str, **kwargs):
        """Create a BinaryFilePart from Base64-encoded binary data."""
        return cls.from_bytes(base64.b64decode(data), mime, **kwargs)

    @classmethod
    def from_b64_uri(cls, data: str):
        if not (prefix_match := re.match("data:(.+);base64,", data)):
            raise ValueError(
                "Data URI must begin with a MIME type indicating Base64 encoding (`data:mime/type;base64,`)."
            )
        return cls.from_b64(data=data[prefix_match.end() :], mime=prefix_match[1])

    @classmethod
    async def from_url(cls, url: str, *, allowed_mime=("*",), **kwargs):
        """
        Download a file from the Internet and create a BinaryFilePart. This saves the data to a temporary file.

        .. attention::
            Note that this classmethod is *asynchronous*, as it downloads data from the web!

        .. tip::
            Certain sites may download all binary data with the ``application/octet-stream`` MIME type. To set the
            MIME type more precisely, use ``mime="..."``.

        Keyword arguments are passed to :meth:`from_file`.
        """
        f = tempfile.TemporaryFile(mode="w+b")
        download_result = await download_media(url, f, allowed_mime=allowed_mime)
        kwargs.setdefault("mime", download_result.mime)
        return cls.from_file(f, **kwargs)

    # ==== representations ====
    def as_bytes(self) -> bytes:
        """Return the full raw data. This could consume a lot of memory!"""
        # this could be slow, but if we're using the same file a lot the OS should put it in RAM cache anyway,
        # and if it runs out of RAM the OS should handle it automatically
        # so we don't need to do memcaching here, thanks OS :)
        self.file.seek(0)
        return self.file.read()

    def as_b64(self) -> str:
        """
        Return the binary data encoded in a base64 string. This could consume a lot of memory!

        Note that this is *not* a web-suitable ``data:mime/...`` string; just the raw binary of the file. Use
        :meth:`as_b64_uri` for a web-suitable string.
        """
        return base64.b64encode(self.as_bytes()).decode()

    def as_b64_uri(self) -> str:
        """Get the binary data encoded in a web-suitable base64 string. This could consume a lot of memory!"""
        return f"data:{self.mime};base64,{self.as_b64()}"

    # ==== helpers ====
    @property
    def _stat(self) -> os.stat_result | None:
        """The file's os.stat info, if available."""
        try:
            # if we have a file descriptor, use os stat
            fileno = self.file.fileno()
            return os.stat(fileno)
        except (io.UnsupportedOperation, AttributeError):
            # otherwise return None
            return None

    @property
    def filesize(self):
        """The size of the file, in bytes."""
        if stat := self._stat:
            return stat.st_size
        # otherwise we'll fall back to seek/tell
        self.file.seek(0, os.SEEK_END)
        return self.file.tell()

    def sha256(self) -> bytes:
        """
        Return the SHA-256 hash of the file contents.

        This method is preferred over manually using ``hashlib.sha256(part.as_bytes())`` as it is speed and
        memory-optimized.
        """
        # if our file is real, we can return cache if it hasn't changed
        if (stat := self._stat) and self._last_hash:
            if stat.st_mtime == self._last_hash_mtime:
                return self._last_hash
            self._last_hash_mtime = stat.st_mtime
        # otherwise,
        # hash reading 64KiB at a time
        self.file.seek(0)
        the_hash = hashlib.sha256()
        while part := self.file.read(65536):
            the_hash.update(part)
        # cache the hash and return
        the_hash_bytes = the_hash.digest()
        self._last_hash = the_hash_bytes
        return the_hash_bytes

    # ==== serdes ====
    @model_serializer()
    def _serialize_binary_file_part(self, info) -> dict[str, str]:
        """
        When we serialize, save the data as:
        - B64 of compressed data when not in zipfile mode
        - a file when in zipfile mode
        """
        if ctx := saveload.get_ctx(info):
            suffix = mimetypes.guess_extension(self.mime) or ""
            fp = ctx.save_bytes(self.as_bytes(), suffix=suffix)
            return {"_archive_path": fp, "mime": self.mime, **self._get_typekey_dict()}
        else:
            compressed_b64 = base64.b64encode(zlib.compress(self.as_bytes())).decode()
            return {"mime": self.mime, "compression": "gzip", "data": compressed_b64, **self._get_typekey_dict()}

    # noinspection PyNestedDecorators
    @model_validator(mode="wrap")
    @classmethod
    def _validate_binary_file_part(cls, v, nxt, info):
        """If the value is the URI we saved, try loading it that way."""
        assert isinstance(v, dict)
        if "_archive_path" in v:
            ctx = saveload.get_ctx(info)
            data = ctx.load_bytes(v["_archive_path"])
            return cls.from_bytes(data, mime=v["mime"])
        elif "data" in v:
            if v.get("compression") == "gzip":
                decompressed = zlib.decompress(base64.b64decode(v["data"]))
                return cls.from_bytes(mime=v["mime"], data=decompressed)
            return cls.from_b64(mime=v["mime"], data=v["data"])
        return nxt(v)

    # ==== lifecycle ====
    def __del__(self):
        self.file.close()


# ==== text ====
class TextPart(BaseMultimodalPart):
    """
    A part representing basic text data.

    Generally you can use a :class:`str` part instead. This part is useful when you need to store additional
    engine-specific metadata alongside a text part.
    """

    text: str

    def __str__(self):
        return self.text
