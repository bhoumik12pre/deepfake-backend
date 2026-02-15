import torch
from torchvision import transforms
from PIL import Image

from model_loader import load_model
from explainability import get_attention_map, overlay_heatmap_on_image

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict_image_pil(image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    label = "Fake" if pred == 0 else "Real"

    confidence = torch.softmax(logits, dim=1)[0][pred].item() * 100

    attn_map = get_attention_map(model, input_tensor)
    heatmap_img = overlay_heatmap_on_image(image, attn_map)

    return label, round(confidence, 2), heatmap_img
