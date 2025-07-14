from pathlib import Path

from kani.ext.multimodal_core.video import VideoPart

from .utils import REPO_ROOT

TEST_VIDEO_PATH = Path(REPO_ROOT / "tests/data/test.mp4")


def test_from_file():
    part = VideoPart.from_file(TEST_VIDEO_PATH)
    assert part.filesize == 8074625
    assert part.duration == 219.080272
    assert part.resolution == (480, 360)
    assert part.mime == "video/mp4"
    # assert part.as_ndarray().shape == (768, 1024, 3)
    # assert part.as_tensor().shape == (3, 768, 1024)


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
