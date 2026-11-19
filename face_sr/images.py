"""PNG transport and uploaded-image decoding shared by the service and API."""

import base64
import io

from PIL import Image, ImageOps


def png_url(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def decode_image(value):
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("Choose a PNG, JPEG, or WebP image.")
    raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Choose an image smaller than 8 MB.")
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Supported image formats are PNG, JPEG, and WebP.")
        if image.width * image.height > 25_000_000:
            raise ValueError("Choose an image smaller than 25 megapixels.")
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba)
        return image.convert("RGB").copy()
