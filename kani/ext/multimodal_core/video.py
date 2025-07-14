import io
import json
import subprocess
from typing import TYPE_CHECKING

import numpy as np

from .base import BinaryFilePart

if TYPE_CHECKING:
    import torch


class VideoPart(BinaryFilePart, arbitrary_types_allowed=True):
    """
    A part representing video data.

    Video data is stored as a file-like object and a MIME type. This allows applications to persist large files on
    disk (using a FileIO) or in memory (using a BytesIO).

    When serialized, video data is represented as a data URI. This can lead to some really big files!

    To get video data in a suitable format for downstream applications, use :meth:`as_b64`, :meth:`as_bytes`,
    :meth:`as_ndarray`, or :meth:`as_tensor`.
    """

    _duration: float = None
    _resolution: tuple[int, int] = None

    # ==== constructors ====
    @classmethod
    async def from_url(cls, url: str, *, allowed_mime=("video/*",), **kwargs):
        """
        Download a video from the Internet and create a VideoPart. This saves the data to a temporary file.

        .. attention::
            Note that this classmethod is *asynchronous*, as it downloads data from the web!

        Keyword arguments are passed to :meth:`from_file`.
        """
        return await super().from_url(url, allowed_mime=allowed_mime, **kwargs)

    # ==== representations ====
    def as_ndarray(self) -> np.ndarray:
        """
        Get the pixel-wise image data as a NumPy array (h*w*c).

        .. warning::

            Note that this array is in (height, width, channels) dimensionality, unlike :meth:`as_tensor` which
            return a tensor in (channels, height, width) dimensionality.
        """
        return np.asarray(self.image)

    def as_tensor(self) -> "torch.Tensor":
        """
        Get the time-pixel-wise video data as a PyTorch tensor (t*c*h*w).

        .. warning::

            Note that this tensor is in (time, channels, height, width) dimensionality, unlike :meth:`as_ndarray` which
            return an array in (height, width, channels) dimensionality.
        """
        try:
            from torchvision.transforms.functional import pil_to_tensor
        except ImportError:
            raise ImportError(
                "PyTorch or torchvision is not installed in your environment. Please install `torch` and `torchvision`"
                " to use `.as_tensor`."
            ) from None

        return pil_to_tensor(self.image)

    # ==== helpers ====
    def _ffprobe(self):
        """Run ffprobe to get the relevant metadata, and cache it"""
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=width,height",
            "-of",
            "json",
            "-",
        ]
        # if we have a file descriptor, pass that to the subprocess instead of reading
        try:
            fileno = self.file.fileno()
            result = subprocess.run(ffprobe_cmd, stdin=fileno, capture_output=True)
        except io.UnsupportedOperation:
            result = subprocess.run(ffprobe_cmd, input=self.as_bytes(), capture_output=True)
        data = json.loads(result.stdout)
        self._duration = float(data["format"]["duration"])
        self._resolution = (int(data["streams"][0]["width"]), int(data["streams"][0]["height"]))

    @property
    def duration(self) -> float:
        """The duration of this video, in seconds."""
        if self._duration is None:
            self._ffprobe()
        return self._duration

    @property
    def resolution(self) -> tuple[int, int]:
        """The resolution of the video's first frame, in pixels (width, height)."""
        if self._resolution is None:
            self._ffprobe()
        return self._resolution
