import torch
from transformers import ViTForImageClassification, ViTConfig
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    config = ViTConfig.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        output_attentions=True
    )

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        config=config,
        ignore_mismatched_sizes=True
    )

    model_path = "model/vit_face_final_best.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found in /model folder")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device
