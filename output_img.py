import os
import uuid
from PIL import Image, ImageDraw, ImageFont
from fastapi import HTTPException
from fastapi.responses import FileResponse


OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_boxes(image: Image.Image, results, model) -> Image.Image:
    """Draw bounding boxes with class name and confidence score on image."""
    draw = ImageDraw.Draw(image)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=18)
    except:
        font = ImageFont.load_default()

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return image

    # Color palette for different classes
    colors = [
        "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
        "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
        "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
        "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7"
    ]

    for box in boxes:
        # Extract box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        class_name = model.names[cls_id]

        # Pick color based on class id
        color = colors[cls_id % len(colors)]

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Prepare label text
        label = f"{class_name} {conf:.2f}"

        # Draw label background
        bbox_text = draw.textbbox((x1, y1), label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill=color)

        # Draw label text
        draw.text((x1 + 3, y1 - text_h - 3), label, fill="white", font=font)

    return image


def save_output_image(image: Image.Image) -> str:
    """Save image to output directory and return the filename."""
    output_filename = f"{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    image.save(output_path, format="JPEG", quality=95)
    return output_filename


def serve_output_image(filename: str) -> FileResponse:
    """Return a FileResponse for the given output image filename."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(file_path, media_type="image/jpeg")