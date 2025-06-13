import math
from pathlib import Path

import soundfile
import torchaudio
from kani.ext.multimodal_core.audio import AudioPart

from .utils import REPO_ROOT

TEST_AUDIO_PATH_MP3 = Path(REPO_ROOT / "tests/data/test.mp3")  # 44.1 kHz
TEST_AUDIO_PATH_WAV = Path(REPO_ROOT / "tests/data/test.wav")  # 24 kHz
TEST_AUDIO_PATH_PCM = Path(REPO_ROOT / "tests/data/test.pcm")  # 24 kHz


def test_from_file_wav():
    audio_part = AudioPart.from_file(TEST_AUDIO_PATH_WAV)
    pcm_bytes = TEST_AUDIO_PATH_PCM.read_bytes()
    assert audio_part.raw == pcm_bytes
    assert audio_part.duration == len(pcm_bytes) / 48000
    assert audio_part.sr == audio_part.sample_rate == 24000


def test_from_file_mp3():
    audio_part = AudioPart.from_file(TEST_AUDIO_PATH_MP3)
    pcm_bytes = TEST_AUDIO_PATH_PCM.read_bytes()
    # downsampling behaviour isn't exactly consistent so we just check that it's +/- 1 frame
    assert math.isclose(audio_part.duration, len(pcm_bytes) / 48000, abs_tol=1 / 24000)


def test_sample_rate_remux():
    audio_part1 = AudioPart(raw=TEST_AUDIO_PATH_PCM.read_bytes(), sample_rate=24000)
    audio_part2 = AudioPart.from_file(TEST_AUDIO_PATH_WAV)
    assert audio_part1.as_bytes(sr=16000) == audio_part2.as_bytes(sr=16000)


def test_numpy_equivalence():
    # load the reference from soundfile
    audio_wav, sr = soundfile.read(TEST_AUDIO_PATH_WAV)
    audio_part = AudioPart(raw=TEST_AUDIO_PATH_PCM.read_bytes(), sample_rate=24000)
    assert (audio_wav == audio_part.as_ndarray(sr=24000)).all()


def test_torch_equivalence():
    # load the reference from torchaudio
    audio_wav, sr = torchaudio.load(TEST_AUDIO_PATH_WAV)
    audio_part = AudioPart(raw=TEST_AUDIO_PATH_PCM.read_bytes(), sample_rate=24000)
    assert (audio_wav == audio_part.as_tensor(sr=24000)).all()


def test_roundtrip_b64():
    audio_part1 = AudioPart.from_file(TEST_AUDIO_PATH_WAV)
    audio_part2 = AudioPart.from_b64(audio_part1.as_b64(sr=24000), sr=24000)
    assert audio_part1.raw == audio_part2.raw


def test_roundtrip_json():
    audio_part1 = AudioPart.from_file(TEST_AUDIO_PATH_WAV)
    audio_part2 = AudioPart.model_validate_json(audio_part1.model_dump_json())
    assert audio_part1.raw == audio_part2.raw
