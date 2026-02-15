from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64

from web_backend import predict_image_pil

app = FastAPI()

# allow frontend requests

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Deepfake API running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    label, confidence, heatmap = predict_image_pil(image)
    
    buf = io.BytesIO()
    heatmap.save(buf, format="PNG")
    heatmap_base64 = base64.b64encode(buf.getvalue()).decode()
    
    return {
        "label": label,
        "confidence": confidence,
        "heatmap": heatmap_base64
    }
