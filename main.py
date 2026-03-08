from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os
import uuid

app = FastAPI()

# Load model once when the server starts
# Ensure 'best.pt' is in the same directory as this script
model = YOLO("best.pt")

@app.post("/analyze")
async def analyze(scan_image: UploadFile = File(...)):
    # 1. Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = f"temp_{file_id}.jpg"
    output_filename = f"ai_analyzed_{file_id}.jpg"
    output_path = f"static/{output_filename}"
    
    os.makedirs("static", exist_ok=True)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(scan_image.file, buffer)

    try:
        # 2. Run Inference
        results = model(input_path, verbose=False)
        
        detections = []
        for r in results:
            r.save(filename=output_path) # Saves processed image to static folder
            for box in r.boxes:
                detections.append({
                    "class_id": int(box.cls),
                    "class_name": r.names[int(box.cls)],
                    "confidence": float(box.conf)
                })

        # 3. Clean up input file
        os.remove(input_path)

        return {
            "success": True,
            "image_url": f"https://your-render-app-name.onrender.com/static/{output_filename}",
            "detections": detections
        }
    except Exception as e:
        return {"success": False, "error": str(e)}