import os
import urllib.request

MODEL_PATH = "model/vit_face_final_best.pth"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1r3o56dhVVqSUd-vOodQHRLlEO_GjqOU7"

os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")
else:
    print("Model already exists.")
