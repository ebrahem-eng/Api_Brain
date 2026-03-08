import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

# 1. Create and Mount the static folder so images are viewable in browser
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 2. Load the model (Ensure best.pt is in the same folder as main.py)
model = YOLO("best.pt")

@app.get("/")
async def root():
    return {"status": "AI Engine Online"}

@app.post("/analyze")
async def analyze(scan_image: UploadFile = File(...)):
    # Create unique filenames
    file_id = str(uuid.uuid4())
    input_path = f"input_{file_id}.jpg"
    output_filename = f"ai_analyzed_{file_id}.jpg"
    output_path = os.path.join(STATIC_DIR, output_filename)

    try:
        # Save the uploaded file locally
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(scan_image.file, buffer)

        # Run YOLO Inference
        results = model(input_path, verbose=False)
        detections = []

        for r in results:
            # This saves the image with the bounding boxes drawn on it
            r.save(filename=output_path)
            for box in r.boxes:
                detections.append({
                    "class_id": int(box.cls),
                    "class_name": r.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "box": box.xyxy[0].tolist() # [x1, y1, x2, y2]
                })

        # Remove the original uploaded file to save space
        if os.path.exists(input_path):
            os.remove(input_path)

        # IMPORTANT: Replace 'api-brain-xxxx' with your actual Render URL
        # You can also use relative paths if Laravel handles the base URL
        return {
            "success": True,
            "image_url": f"https://api-brain-xxxx.onrender.com/static/{output_filename}",
            "detections": detections
        }

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        return {"success": False, "error": str(e)}