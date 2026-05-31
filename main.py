from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from collections import Counter
from PIL import Image
import io

from output_img import draw_boxes, save_output_image, serve_output_image


app = FastAPI(
    title="mankind-healthok API"
)

posm_mapping = {
    "posm1": "healthok_back",
    "posm2": "healthok_front",
    "posm3": "men_healthok",
    "posm4": "women_healthok"
}

# All required class values that must be detected for a valid result
REQUIRED_CLASSES = set(posm_mapping.values())


def validate_posm(detected_class_names: list[str]) -> dict:
    """
    Check if all required POSM classes are present in detections.

    Returns a dict with:
      - is_valid (bool): True if all POSMs detected
      - missing_posms (list): POSM keys whose classes were not detected
    """
    detected_set = set(detected_class_names)

    missing_posms = [
        class_name
        for posm_key, class_name in posm_mapping.items()
        if class_name not in detected_set
    ]

    return {
        "is_valid": len(missing_posms) == 0,
        "missing_posms": missing_posms
    }


try:
    model = YOLO("mankind_29may.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e


@app.post("/detect")
async def detect_and_count(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        results = model.predict(
            source=image,
            conf=0.5,
            iou=0.5,
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            output_filename = save_output_image(image)

            # No detections — all POSMs are missing
            validation = validate_posm([])

            return {
                "filename": file.filename,
                "total_objects": 0,
                "counts": {},
                "detections": [],
                "output_pic": f"/output/{output_filename}",
                "status": "invalid",
                "missing_posms": validation["missing_posms"]
            }

        # Extract detection details
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy().tolist()
        xyxy_boxes = boxes.xyxy.cpu().numpy().tolist()
        detected_names = [model.names[i] for i in class_ids]

        # Build per-detection list
        detections = [
            {
                "class": detected_names[i],
                "confidence": round(confidences[i], 4),
                "bbox": {
                    "x1": int(xyxy_boxes[i][0]),
                    "y1": int(xyxy_boxes[i][1]),
                    "x2": int(xyxy_boxes[i][2]),
                    "y2": int(xyxy_boxes[i][3])
                }
            }
            for i in range(len(detected_names))
        ]

        # Draw boxes and save annotated image
        annotated_image = draw_boxes(image.copy(), results, model)
        output_filename = save_output_image(annotated_image)

        counts = dict(Counter(detected_names))

        # Validate POSM presence
        validation = validate_posm(detected_names)

        response = {
            "filename": file.filename,
            "total_objects": len(detected_names),
            "counts": counts,
            "output_pic": f"/output/{output_filename}",
            "status": "valid" if validation["is_valid"] else "invalid",
        }

        # Only include missing_posms in response if invalid
        if not validation["is_valid"]:
            response["missing_posms"] = validation["missing_posms"]

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/output/{filename}")
async def get_output_image(filename: str):
    """Serve the annotated output image by filename."""
    return serve_output_image(filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)