"""PatchCore feature extractor - DINOv2 (ViT-S/14) backbone.

Swapped from WideResNet50-2 to DINOv2 because DINOv2's self-supervised dense
patch features are color/appearance sensitive (not texture-biased), which
greatly improves discoloration detection (recall 0.57 -> 0.90) and, on real
cracks, reaches ~100%.

The class name `Extractor` and the `score_with_map(pil, bank)` signature are
kept identical to the previous WideResNet version, so the serving code
(APP_fastapi.py: `from patchcore import Extractor`) works unchanged.

Banks in models/patchcore_200/<SKU>.npz are DINOv2 memory banks (key "bank")
with a p99 threshold (key "threshold"). DINOv2 weights are fetched at runtime
via torch.hub (facebookresearch/dinov2) - internet required on first run.
"""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]


class Extractor:
    def __init__(self, name="dinov2_vits14", input=224):
        self.net = torch.hub.load("facebookresearch/dinov2", name).to(DEVICE).eval()
        self.input = input
        self.patch = 14
        self.tf = T.Compose([
            T.Resize((input, input)),
            T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD),
        ])

    @torch.no_grad()
    def score_with_map(self, pil, bank):
        """Anomaly map + scalar score. score = max over patch tokens of the
        token's nearest-neighbour distance to the memory bank."""
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(pil)
        x = self.tf(pil.convert("RGB")).unsqueeze(0).to(DEVICE)
        tokens = self.net.forward_features(x)["x_norm_patchtokens"][0]   # (N, C)
        if not torch.is_tensor(bank):
            bank = torch.tensor(bank, dtype=torch.float32)
        bank = bank.to(tokens.device, dtype=tokens.dtype)
        d = torch.cdist(tokens, bank).min(dim=1).values                 # (N,)
        g = self.input // self.patch
        grid = d.reshape(g, g).cpu().numpy()
        return grid, float(d.max().item())
