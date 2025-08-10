import base64
import io
from typing import Optional
from PIL import Image
import textwrap

def read_text_file(b: bytes) -> str:
    return b.decode('utf-8')

def image_to_data_uri(img: Image.Image, fmt: str='PNG', max_bytes: int=100000) -> str:
    """
    Encode PIL image to a data URI (base64) and try to compress/resize until < max_bytes.
    fmt: 'PNG' or 'WEBP' recommended.
    """
    buf = io.BytesIO()

    # attempt to save with decreasing quality / size
    quality = 95
    width, height = img.size
    while True:
        buf.seek(0)
        buf.truncate(0)
        if fmt.upper() in ['JPEG','JPG']:
            img.save(buf, format='JPEG', quality=quality, optimize=True)
        else:
            # prefer WEBP for smaller sizes if supported
            img.save(buf, format=fmt, quality=quality, method=6)

        data = buf.getvalue()
        if len(data) <= max_bytes or (width < 200 and height < 200):
            break

        # reduce quality/size
        quality = max(20, quality - 15)
        width = max(200, int(width * 0.85))
        height = max(200, int(height * 0.85))
        img = img.resize((width, height), Image.LANCZOS)

    b64 = base64.b64encode(data).decode('ascii')
    mime = 'image/png' if fmt.upper() == 'PNG' else 'image/webp'
    return f"data:{mime};base64,{b64}"

def ensure_small_image_bytes(b: bytes, max_bytes: int=100000, fmt: str='WEBP') -> str:
    from PIL import Image
    img = Image.open(io.BytesIO(b)).convert('RGBA')
    return image_to_data_uri(img, fmt=fmt, max_bytes=max_bytes)
