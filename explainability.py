import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch

def get_attention_map(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)
        attn = outputs.attentions[-1].mean(dim=1)[0]
        cls_attn = attn[0, 1:]

        grid_size = int(cls_attn.size(0) ** 0.5)
        cls_attn = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())

        return cls_attn

def overlay_heatmap_on_image(image, heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size)
    heatmap_np = np.array(heatmap)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.imshow(heatmap_np, cmap="jet", alpha=0.5)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)

