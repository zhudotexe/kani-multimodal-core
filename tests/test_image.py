from pathlib import Path

import pytest
from kani import ChatMessage, Kani
from kani.ext.multimodal_core.image import ImagePart

from .utils import REPO_ROOT

TEST_IMAGE_PATH = Path(REPO_ROOT / "tests/data/test.png")  # 1024 x 768


def test_from_file():
    part = ImagePart.from_file(TEST_IMAGE_PATH)
    assert part.size == (1024, 768)
    assert part.mime == "image/png"
    assert part.as_ndarray().shape == (768, 1024, 3)
    assert part.as_tensor().shape == (3, 768, 1024)


def test_roundtrip_b64():
    part1 = ImagePart.from_file(TEST_IMAGE_PATH)
    part2 = ImagePart.from_b64(part1.as_b64())
    assert part1.as_bytes() == part2.as_bytes()


def test_roundtrip_json():
    part1 = ImagePart.from_file(TEST_IMAGE_PATH)
    part2 = ImagePart.model_validate_json(part1.model_dump_json())
    assert part1.as_bytes() == part2.as_bytes()


def test_sha256():
    part1 = ImagePart.from_file(TEST_IMAGE_PATH)
    part2 = ImagePart.from_file(TEST_IMAGE_PATH)
    assert part1.sha256()
    assert part1.sha256() == part2.sha256()


@pytest.mark.parametrize("save_format", ("json", "kani"))
def test_saveload(save_format, tmp_path, dummy_engine):
    ai = Kani(dummy_engine, chat_history=[ChatMessage.user([ImagePart.from_file(TEST_IMAGE_PATH)])])

    # save and load
    ai.save(tmp_path / f"pytest.{save_format}", save_format=save_format)
    loaded = Kani(dummy_engine)
    loaded.load(tmp_path / f"pytest.{save_format}")

    # assert equality
    assert ai.chat_history[0].parts[0].as_bytes() == loaded.chat_history[0].parts[0].as_bytes()
