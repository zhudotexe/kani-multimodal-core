from pathlib import Path

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
