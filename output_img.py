import os
import io
import uuid
import time
import jwt
from PIL import Image, ImageDraw, ImageFont
from fastapi.responses import RedirectResponse
from supabase import create_client, Client


# ── Supabase setup ────────────────────────────────────────────────────────────
SUPABASE_URL        = os.environ["SUPABASE_URL"]
SUPABASE_JWT_SECRET = os.environ["SUPABASE_JWT_SECRET"]  # Legacy JWT secret
BUCKET_NAME         = os.environ.get("SUPABASE_BUCKET", "output-images")

# Generate a valid service_role JWT from the legacy secret
_service_role_jwt = jwt.encode(
    {"role": "service_role", "iss": "supabase", "iat": int(time.time()), "exp": int(time.time()) + 315360000},
    SUPABASE_JWT_SECRET,
    algorithm="HS256"
)

supabase: Client = create_client(SUPABASE_URL, _service_role_jwt)


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_boxes(image: Image.Image, results, model) -> Image.Image:
    """Draw bounding boxes with class name and confidence score on image."""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return image

    colors = [
        "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
        "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
        "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
        "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7"
    ]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id     = int(box.cls[0].cpu().numpy())
        conf       = float(box.conf[0].cpu().numpy())
        class_name = model.names[cls_id]
        color      = colors[cls_id % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label     = f"{class_name} {conf:.2f}"
        bbox_text = draw.textbbox((x1, y1), label, font=font)
        text_w    = bbox_text[2] - bbox_text[0]
        text_h    = bbox_text[3] - bbox_text[1]
        draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - text_h - 3), label, fill="white", font=font)

    return image


# ── Storage helpers ───────────────────────────────────────────────────────────
def save_output_image(image: Image.Image) -> str:
    """Upload annotated image to Supabase Storage. Returns the storage path."""
    output_filename = f"{uuid.uuid4().hex}.jpg"

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    image_bytes = buf.getvalue()

    supabase.storage.from_(BUCKET_NAME).upload(
        path=output_filename,
        file=image_bytes,
        file_options={"content-type": "image/jpeg"}
    )

    return output_filename


def get_public_url(storage_path: str) -> str:
    """Return the public URL for a stored image."""
    return supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)


def serve_output_image(filename: str) -> RedirectResponse:
    """Redirect /output/{filename} to the Supabase public URL."""
    public_url = get_public_url(filename)
    return RedirectResponse(url=public_url)