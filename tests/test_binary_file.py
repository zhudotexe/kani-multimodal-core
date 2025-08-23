import hashlib
import io
import tempfile
import zipfile
from pathlib import Path

import pytest
from kani import ChatMessage, Kani
from kani.ext.multimodal_core.base import BinaryFilePart

from .utils import REPO_ROOT

TEST_FILE_PATH = Path(REPO_ROOT / "tests/data/test.pdf")


def test_from_file():
    part = BinaryFilePart.from_file(TEST_FILE_PATH)
    assert part.filesize == 1273097
    assert part.mime == "application/pdf"
    assert part.as_bytes() == TEST_FILE_PATH.read_bytes()


async def test_from_url():
    part = await BinaryFilePart.from_url(
        "https://raw.githubusercontent.com/zhudotexe/kani-multimodal-core/main/tests/data/test.pdf",
        mime="application/pdf",
    )
    assert part.filesize == 1273097
    assert part.as_bytes() == TEST_FILE_PATH.read_bytes()
    assert part.mime == "application/pdf"


def test_file_like_io():
    part = BinaryFilePart.from_file(io.BytesIO(b"hello world"), mime="text")
    assert part.filesize == 11
    assert part.mime == "text"


def test_file_like_zip():
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".zip") as zft:
        # write some zip data
        with zipfile.ZipFile(zft, mode="w") as zf:
            zf.write(TEST_FILE_PATH, "test.pdf")
        zft.seek(0)

        # then ensure we can make a filepart from it
        with zipfile.ZipFile(zft, mode="r") as zf:
            with zf.open("test.pdf") as f:
                part = BinaryFilePart.from_file(f)
                assert part.filesize == 1273097
                assert part.as_bytes() == TEST_FILE_PATH.read_bytes()
                assert part.mime == "application/pdf"


def test_roundtrip_b64():
    part1 = BinaryFilePart.from_file(TEST_FILE_PATH)
    part2 = BinaryFilePart.from_b64(part1.as_b64(), mime=part1.mime)
    assert part1.as_bytes() == part2.as_bytes()


def test_roundtrip_b64_uri():
    part1 = BinaryFilePart.from_file(TEST_FILE_PATH)
    part2 = BinaryFilePart.from_b64_uri(part1.as_b64_uri())
    assert part1.as_bytes() == part2.as_bytes()


def test_roundtrip_json():
    part1 = BinaryFilePart.from_file(TEST_FILE_PATH)
    part2 = BinaryFilePart.model_validate_json(part1.model_dump_json())
    assert part1.as_bytes() == part2.as_bytes()


def test_sha256():
    part1 = BinaryFilePart.from_file(TEST_FILE_PATH)
    assert part1.sha256()
    assert part1.sha256() == hashlib.sha256(TEST_FILE_PATH.read_bytes()).digest()


@pytest.mark.parametrize("save_format", ("json", "kani"))
def test_saveload(save_format, tmp_path, dummy_engine):
    ai = Kani(dummy_engine, chat_history=[ChatMessage.user([BinaryFilePart.from_file(TEST_FILE_PATH)])])

    # save and load
    ai.save(tmp_path / f"pytest.{save_format}", save_format=save_format)
    loaded = Kani(dummy_engine)
    loaded.load(tmp_path / f"pytest.{save_format}")

    # assert equality
    assert ai.chat_history[0].parts[0].as_bytes() == loaded.chat_history[0].parts[0].as_bytes()
