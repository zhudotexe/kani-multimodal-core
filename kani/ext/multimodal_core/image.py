import base64
import io
import mimetypes
import re
from typing import IO, TYPE_CHECKING

import numpy as np
from PIL import Image
from kani.utils.typing import PathLike
from pydantic import model_serializer, model_validator

from .base import BaseMultimodalPart
from .utils import download_media

if TYPE_CHECKING:
    import torch


class ImagePart(BaseMultimodalPart, arbitrary_types_allowed=True):
    """
    A part representing image data.

    Image data is stored in memory as a Pillow Image object.
    When serialized, image data is represented as a data URI.

    To get audio data in a suitable format for downstream applications, use :meth:`as_b64`, :meth:`as_bytes`,
    :meth:`as_ndarray`, or :meth:`as_tensor`.
    """

    image: Image.Image

    # ==== constructors ====
    @classmethod
    def from_file(cls, fp: PathLike | IO, **kwargs):
        """
        Create an ImagePart from a local image file. The file format will be automatically detected.
        """
        return cls(image=Image.open(fp), **kwargs)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs):
        """Create an ImagePart from raw binary data."""
        return cls(image=Image.open(io.BytesIO(data)), **kwargs)

    @classmethod
    def from_b64(cls, data: str, **kwargs):
        """Create an ImagePart from Base64-encoded binary data."""
        return cls.from_bytes(base64.b64decode(data), **kwargs)

    @classmethod
    def from_b64_uri(cls, data: str):
        if not (match := re.match("data:(image/.+);base64,(.*)", data)):
            raise ValueError("Data URI must begin with an image MIME type (`data:image/*;base64,`)")
        mime = match[1]
        extensions = [e.removeprefix(".") for e in mimetypes.guess_all_extensions(mime, strict=False)]
        img_bytes = base64.b64decode(match[2])
        return cls.from_bytes(img_bytes, formats=extensions or None)

    @classmethod
    async def from_url(cls, url: str, **kwargs):
        """
        Download an image from the Internet and create an ImagePart.

        .. attention::
            Note that this classmethod is *asynchronous*, as it downloads data from the web!

        Keyword arguments are passed to :meth:`from_file`.
        """
        f = io.BytesIO()
        await download_media(url, f, allowed_mime=("image/*",))
        return cls.from_file(f, **kwargs)

    # ==== representations ====
    def as_bytes(self, format: str = "png") -> bytes:
        """Return the raw image data in the given format."""
        f = io.BytesIO()
        self.image.save(f, format=format)
        return f.getvalue()

    def as_b64(self, format: str = "png") -> str:
        """
        Return the binary image data in the given format encoded in a base64 string.

        Note that this is *not* a web-suitable ``data:image/...`` string; just the raw binary of the image. Use
        :meth:`as_b64_uri` for a web-suitable string.
        """
        return base64.b64encode(self.as_bytes(format)).decode()

    def as_b64_uri(self, format: str = "png") -> str:
        """Get the binary image data encoded in a web-suitable base64 string."""
        format = format.lower()
        mime = Image.MIME.get(format, mimetypes.types_map.get(f".{format}", f"image/{format}"))
        return f"data:{mime};base64,{self.as_b64(format)}"

    def as_ndarray(self) -> np.ndarray:
        """
        Get the pixel-wise image data as a NumPy array (h*w*c).

        .. warning::

            Note that this array is in (height, width, channels) dimensionality, unlike :meth:`to_tensor` which
            return a tensor in (channels, height, width) dimensionality.
        """
        return np.asarray(self.image)

    def as_tensor(self) -> "torch.Tensor":
        """
        Get the pixel-wises image data as a PyTorch tensor (c*h*w).

        .. warning::

            Note that this tensor is in (channels, height, width) dimensionality, unlike :meth:`to_ndarray` which
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
    @property
    def size(self) -> tuple[int, int]:
        """The size of the image, in pixels (width, height)."""
        return self.image.size

    @property
    def mime(self) -> str:
        """The MIME filetype of the image."""
        img_format = self.image.format
        return Image.MIME.get(
            img_format, mimetypes.types_map.get(f".{img_format.lower()}", f"image/{img_format.lower()}")
        )

    # ==== serdes ====
    @model_serializer(when_used="json")
    def _serialize_imagepart(self) -> dict[str, str]:
        """When we serialize to JSON, save the data as a URI"""
        return {"img_data": self.as_b64_uri()}

    # noinspection PyNestedDecorators
    @model_validator(mode="wrap")
    @classmethod
    def _validate_imagepart(cls, v, nxt):
        """If the value is the URI we saved, try loading it that way"""
        if isinstance(v, dict) and "img_data" in v:
            return cls.from_b64_uri(v["img_data"])
        return nxt(v)
