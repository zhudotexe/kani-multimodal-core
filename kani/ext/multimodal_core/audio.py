"""Core MessageParts for Kani multimodal"""

import base64
import hashlib
import io
import warnings
import wave
from typing import IO, TYPE_CHECKING

import numpy as np
from kani.utils import saveload
from kani.utils.typing import PathLike
from pydantic import Field, model_serializer, model_validator
from pydub import AudioSegment

from .base import BaseMultimodalPart
from .utils import download_media

if TYPE_CHECKING:
    import torch


class AudioPart(BaseMultimodalPart):
    """
    A part representing audio data.

    Audio data is stored in memory as raw signed 16-bit little-endian mono PCM in :attr:`raw`,
    at a variable :attr:`sample_rate`.
    When serialized, audio data is represented as a data URI.

    To get audio data in a suitable format for downstream applications, use :meth:`as_b64`, :meth:`as_bytes`,
    :meth:`as_ndarray`, or :meth:`as_tensor`.
    """

    raw: bytes = Field(repr=False)
    """The raw binary data in signed 16-bit little-endian mono PCM format."""

    sample_rate: int
    """The sample rate of the binary data."""

    # ==== constructors ====
    @classmethod
    def from_b64(cls, data: str, sr: int, **kwargs):
        """Create an AudioPart from Base64-encoded signed 16-bit little-endian mono PCM data."""
        return cls(raw=base64.b64decode(data), sample_rate=sr, **kwargs)

    @classmethod
    def from_file(
        cls,
        fp: PathLike | IO,
        *,
        format: str = None,
        codec: str = None,
        converter_parameters: str = None,
        sr: int = None,
        sample_width: int = None,
        channels: int = None,
        **kwargs,
    ):
        """
        Create an AudioPart from a local file.

        :param fp: The path to the file or an open file to read.
        :param format: The format (e.g. 'mp3') of the audio file. Will attempt to automatically determine based on the
            given filename if this is not set.
        :param codec: An explicit audio codec to use to decode the audio file, if conversion is needed. (See FFMPEG's
            ``-acodec`` option for valid inputs).
        :param converter_parameters: Any additional CLI arguments to pass to the audio converter, if conversion is
            needed.
        :param sr: The sample rate of the audio (raw PCM audio only).
        :param sample_width: The sample width, in bytes, of the audio (raw PCM audio only).
        :param channels: The number of channels of the audio (raw PCM audio only).
        """
        segment = AudioSegment.from_file(
            fp,
            format=format,
            codec=codec,
            parameters=converter_parameters,
            frame_rate=sr,
            sample_width=sample_width,
            channels=channels,
        )
        mono = segment.set_channels(1).set_sample_width(2)
        return cls(raw=mono.raw_data, sample_rate=mono.frame_rate, **kwargs)

    @classmethod
    def from_wav_b64_uri(cls, data: str):
        if not data.startswith("data:audio/wav;base64,"):
            raise ValueError("Data URI must begin with `data:audio/wav;base64,`")
        wav_bytes = base64.b64decode(data.removeprefix("data:audio/wav;base64,"))
        return cls.from_file(io.BytesIO(wav_bytes), format="wav")

    @classmethod
    async def from_url(cls, url: str, **kwargs):
        """
        Download audio from the Internet and create an AudioPart.

        .. attention::
            Note that this classmethod is *asynchronous*, as it downloads data from the web!

        Keyword arguments are passed to :meth:`from_file`.
        """
        f = io.BytesIO()
        await download_media(url, f, allowed_mime=("audio/*",))
        return cls.from_file(f, **kwargs)

    # ==== representations ====
    # --- raw ---
    def _as_bytes(self, sr: int = None) -> bytes:
        # this is a private method so the warning's stacklevel is correct
        if sr is None:
            warnings.warn(
                "AudioPart.as_bytes() was called with no explicit sample rate given. Returning at the raw sample rate"
                f" of {self.sample_rate}! Pass `sr=...` or use `.resample()` to use a different rate.",
                stacklevel=3,
            )
            return self.raw
        if sr == self.sample_rate:
            return self.raw
        # sample to the specified sr and return
        segment = AudioSegment(self.raw, sample_width=2, frame_rate=self.sample_rate, channels=1)
        return segment.set_frame_rate(sr).raw_data

    def as_bytes(self, sr: int = None) -> bytes:
        """Return the audio data as signed 16-bit little-endian mono PCM at the given sample rate."""
        return self._as_bytes(sr)

    def as_b64(self, sr: int = None) -> str:
        """Return the audio data as Base64-encoded signed 16-bit little-endian mono PCM at the given sample rate."""
        return base64.b64encode(self._as_bytes(sr)).decode()

    def as_ndarray(self, sr: int = None) -> np.ndarray:
        """Return the audio data as a 1-dimensional NumPy array of floats at the given sample rate."""
        # equivalence verify
        # $ ffmpeg -i test.mp3 -ac 1 -ar 24000 test.wav
        # $ ffmpeg -i test.mp3 -f s16le -acodec pcm_s16le -ac 1 -ar 24000 test.pcm
        # import soundfile, numpy as np
        # from pathlib import Path
        # audio_path_wav = Path("test.wav")
        # audio_path_pcm = Path("test.pcm")
        # audio_wav, sr = soundfile.read(audio_path_wav)
        # audio_bytes = audio_path_pcm.read_bytes()
        # audio_ints = np.frombuffer(audio_bytes, dtype=np.int16)
        # audio_wav2 = audio_ints / 32768
        # (audio_wav == audio_wav2).all()
        audio_ints = np.frombuffer(self._as_bytes(sr), dtype=np.int16)
        return audio_ints / 32768

    def as_tensor(self, sr: int = None) -> "torch.Tensor":
        """
        Return the audio data as a 2-dimensional [channel, time] PyTorch Tensor of floats at the given sample rate.

        Note that since this library only uses mono audio, that the first dimension will always be 1.
        """
        # equivalence verify
        # $ ffmpeg -i test.mp3 -ac 1 -ar 24000 test.wav
        # $ ffmpeg -i test.mp3 -f s16le -acodec pcm_s16le -ac 1 -ar 24000 test.pcm
        # import torchaudio, torch
        # from pathlib import Path
        # audio_path_pcm = Path("test.pcm")
        # audio_wav, sr = torchaudio.load("test.wav")
        # audio_bytes = audio_path_pcm.read_bytes()
        # audio_ints = torch.frombuffer(audio_bytes, dtype=torch.int16)
        # audio_wav2 = audio_ints.div(32768).reshape(1, -1)
        # (audio_wav == audio_wav2).all()
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is not installed in your environment. Please install `torch` to use `.as_tensor`."
            ) from None

        audio_ints = torch.frombuffer(self._as_bytes(sr), dtype=torch.int16)
        return audio_ints.div(32768).reshape(1, -1)

    # --- WAV ---
    def as_wav_bytes(self) -> bytes:
        """Return the audio data as WAV data (including header)."""
        out_bytes = io.BytesIO()
        with wave.open(out_bytes, "wb") as wave_data:
            wave_data.setnchannels(1)
            wave_data.setsampwidth(2)
            wave_data.setframerate(self.sample_rate)
            wave_data.setnframes(len(self.raw) // 2)
            wave_data.writeframesraw(self.raw)

        out_bytes.seek(0)
        return out_bytes.getvalue()

    def as_wav_b64_uri(self) -> str:
        """Return the WAV audio data encoded in a web-suitable base64 string."""
        wav_b64 = base64.b64encode(self.as_wav_bytes()).decode()
        return f"data:audio/wav;base64,{wav_b64}"

    # ==== helpers ====
    def resample(self, sample_rate: int) -> "AudioPart":
        """Return a new AudioPart with the given sample rate."""
        if sample_rate == self.sample_rate:
            return self
        return AudioPart(raw=self._as_bytes(sr=sample_rate), sample_rate=sample_rate)

    def sha256(self) -> bytes:
        """Return the SHA-256 hash of the raw audio."""
        return hashlib.sha256(self.raw).digest()

    @property
    def duration(self) -> float:
        """The duration of this audio clip, in seconds."""
        # 16b mono -> 2 bytes per sample * sample rate
        return len(self.raw) / (self.sample_rate * 2)

    @property
    def sr(self):
        """An alias to :attr:`sample_rate`."""
        return self.sample_rate

    @sr.setter
    def sr(self, value):
        self.sample_rate = value

    def __repr__(self):
        audio_repr = f"[audio: {self.duration:.3f}s]"
        return f'{self.__repr_name__()}({self.__repr_str__(", ")}, raw={audio_repr})'

    def __rich_repr__(self):
        audio_repr = f"[audio: {self.duration:.3f}s]"
        yield "raw", audio_repr

    # ==== serdes ====
    @model_serializer()
    def _serialize_audiopart(self, info) -> dict[str, str]:
        """
        When we serialize to JSON, save the data as:
        - a URI when not in zipfile mode
        - a WAV file when in zipfile mode
        """
        if ctx := saveload.get_ctx(info):
            fp = ctx.save_bytes(self.as_wav_bytes(), suffix=".wav")
            return {"_archive_path": fp, **self._get_typekey_dict()}
        else:
            return {"wav_data": self.as_wav_b64_uri(), **self._get_typekey_dict()}

    # noinspection PyNestedDecorators
    @model_validator(mode="wrap")
    @classmethod
    def _validate_audiopart(cls, v, nxt, info):
        """If the value is the URI we saved, try loading it that way"""
        assert isinstance(v, dict)
        if "_archive_path" in v:
            ctx = saveload.get_ctx(info)
            wav_data = ctx.load_bytes(v["_archive_path"])
            return cls.from_file(io.BytesIO(wav_data), format="wav")
        elif "wav_data" in v:
            return cls.from_wav_b64_uri(v["wav_data"])
        return nxt(v)
