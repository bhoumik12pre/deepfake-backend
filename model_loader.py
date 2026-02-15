import torch
from transformers import ViTForImageClassification

_model = None

def get_model():
global _model

```
if _model is None:
    print("Loading model into memory...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2
    )

    state = torch.load("model/vit_face_final_best.pth", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    _model = model

return _model
```
