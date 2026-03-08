import uuid
import os
import shutil
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn

app = FastAPI()

MODEL_PATH = "./weights/best.pt"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
    else:
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

@app.get("/")
def health():
    return {"status": "YOLO API is running", "numpy": np.__version__}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    temp_id = str(uuid.uuid4())
    temp_path = f"/tmp/{temp_id}_{file.filename}"
    output_dir = "/tmp/ai_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = None

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        results = model(temp_path, verbose=False)
        output_filename = f"ai_analyzed_{temp_id}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        detections = []
        for r in results:
            r.save(filename=output_path)
            for box in r.boxes:
                detections.append({
                    "class_id": int(box.cls),
                    "class_name": r.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "coordinates": box.xyxy[0].tolist()
                })

        with open(output_path, "rb") as img_f:
            img_base64 = base64.b64encode(img_f.read()).decode("utf-8")

        return JSONResponse({
            "success": True,
            "image_base64": img_base64,
            "detections": detections
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if output_path and os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)




