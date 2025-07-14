from pathlib import Path

from kani.ext.multimodal_core.base import BinaryFilePart

from .utils import REPO_ROOT

TEST_FILE_PATH = Path(REPO_ROOT / "tests/data/test.pdf")


def test_from_file():
    part = BinaryFilePart.from_file(TEST_FILE_PATH)
    assert part.filesize == 1273097
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
