import hashlib
from pathlib import Path

import pytest
from kani import ChatMessage, Kani
from kani.ext.multimodal_core.video import VideoPart

from .utils import REPO_ROOT

TEST_VIDEO_PATH = Path(REPO_ROOT / "tests/data/test.mp4")


def test_from_file():
    part = VideoPart.from_file(TEST_VIDEO_PATH)
    assert part.filesize == 9122437
    assert part.duration == 219.099
    assert part.resolution == (480, 360)
    assert part.mime == "video/mp4"
    assert part.as_tensor().shape == (220, 3, 360, 480)


def test_roundtrip_b64():
    part1 = VideoPart.from_file(TEST_VIDEO_PATH)
    part2 = VideoPart.from_b64(part1.as_b64(), mime=part1.mime)
    assert part1.as_bytes() == part2.as_bytes()


def test_roundtrip_b64_uri():
    part1 = VideoPart.from_file(TEST_VIDEO_PATH)
    part2 = VideoPart.from_b64_uri(part1.as_b64_uri())
    assert part1.as_bytes() == part2.as_bytes()


def test_roundtrip_json():
    part1 = VideoPart.from_file(TEST_VIDEO_PATH)
    part2 = VideoPart.model_validate_json(part1.model_dump_json())
    assert part1.as_bytes() == part2.as_bytes()


def test_sha256():
    part1 = VideoPart.from_file(TEST_VIDEO_PATH)
    assert part1.sha256()
    assert part1.sha256() == hashlib.sha256(TEST_VIDEO_PATH.read_bytes()).digest()


@pytest.mark.parametrize("save_format", ("json", "kani"))
def test_saveload(save_format, tmp_path, dummy_engine):
    ai = Kani(dummy_engine, chat_history=[ChatMessage.user([VideoPart.from_file(TEST_VIDEO_PATH)])])

    # save and load
    ai.save(tmp_path / f"pytest.{save_format}", save_format=save_format)
    loaded = Kani(dummy_engine)
    loaded.load(tmp_path / f"pytest.{save_format}")

    # assert equality
    assert ai.chat_history[0].parts[0].as_bytes() == loaded.chat_history[0].parts[0].as_bytes()
