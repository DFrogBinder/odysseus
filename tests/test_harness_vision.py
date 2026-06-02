import base64

import pytest

from src.codex_harness.vision import image_path_to_content_block


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_png_file_becomes_openai_image_url_block(tmp_path):
    image = tmp_path / "pixel.png"
    image.write_bytes(PNG_1X1)

    block = image_path_to_content_block(image)

    assert block["type"] == "image_url"
    url = block["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG_1X1


@pytest.mark.parametrize(
    ("name", "payload", "mime"),
    [
        ("photo.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 16, "image/jpeg"),
        ("photo.webp", b"RIFF" + (b"\x00" * 4) + b"WEBPVP8 " + (b"\x00" * 8), "image/webp"),
    ],
)
def test_jpeg_and_webp_are_supported(tmp_path, name, payload, mime):
    image = tmp_path / name
    image.write_bytes(payload)

    block = image_path_to_content_block(image)

    assert block["image_url"]["url"].startswith(f"data:{mime};base64,")


def test_non_image_file_is_rejected(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError):
        image_path_to_content_block(text)


def test_oversized_image_file_is_rejected(tmp_path):
    image = tmp_path / "big.png"
    image.write_bytes(PNG_1X1)

    with pytest.raises(ValueError):
        image_path_to_content_block(image, max_bytes=4)
