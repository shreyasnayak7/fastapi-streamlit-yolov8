from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import torch, shutil, uuid, os
import ultralytics.nn.tasks as yolo_tasks
from PIL import Image

# Allow DetectionModel global
torch.serialization.add_safe_globals([yolo_tasks.DetectionModel])

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Serve results folder
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

# Load YOLO model
MODEL_PATH = "yolov8m.pt"
model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded image
        file_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run YOLO detection
        results = model.predict(input_path, classes=[0])

        # Save annotated image
        result_image_path = os.path.join(RESULT_DIR, f"{file_id}.jpg")
        img = results[0].plot()  # returns a NumPy array
        im = Image.fromarray(img)
        im.save(result_image_path)

        # Return JSON
        return {"result_image": f"results/{file_id}.jpg"}

    except Exception as e:
        return {"error": str(e)}
