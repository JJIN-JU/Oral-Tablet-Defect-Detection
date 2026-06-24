"""Compact PatchCore (from scratch, torchvision backbone) trained per SKU on
normal images only. Methodology matches PatchCore: WideResNet50 layer2+layer3
locally-aware patch features -> greedy coreset memory bank -> kNN anomaly score.

Constraint: NORMAL DATA ONLY. No synthetic defects are generated.
To still produce a real number, we add a cross-SKU SEPARABILITY proxy:
score a SKU's own val (label 0) against a sample of OTHER SKUs' images
(label 1) and compute AUROC. This is NOT defect detection -- it only shows the
memory bank captures "this pill's normal appearance" vs a different pill.

Outputs:
  runs/patchcore/<SKU>.npz          memory bank + threshold
  runs/patchcore_metrics.json       per-SKU + aggregate metrics
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

DATASET = r"F:\HumanAI\dataset"
RUNS = r"F:\HumanAI\pill_pipeline\runs"
SPLIT_JSON = os.path.join(RUNS, "split.json")
BANK_DIR = os.path.join(RUNS, "patchcore")
OUT_JSON = os.path.join(RUNS, "patchcore_metrics.json")

IMG = 224
CORESET = 2048          # memory bank size cap per SKU
CAND_CAP = 16384        # candidate cap before greedy coreset
N_NEG_SKUS = 20         # other SKUs sampled as proxy anomalies
N_NEG_PER = 2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Extractor:
    def __init__(self):
        w = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        net = wide_resnet50_2(weights=w).to(DEVICE).eval()
        self.tf = w.transforms()
        self.feats = {}
        net.layer2.register_forward_hook(self._hook("l2"))
        net.layer3.register_forward_hook(self._hook("l3"))
        self.net = net

    def _hook(self, k):
        def fn(m, i, o):
            self.feats[k] = o
        return fn

    @torch.no_grad()
    def patches(self, pil_list):
        x = torch.stack([self.tf(im) for im in pil_list]).to(DEVICE)
        self.feats.clear()
        self.net(x)
        l2, l3 = self.feats["l2"], self.feats["l3"]
        # locally aware (3x3 avg) aggregation
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.avg_pool2d(l3, 3, 1, 1)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([l2, l3], dim=1)                  # (B,C,H,W)
        b, c, h, w = f.shape
        self.last_hw = (h, w)
        return f.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, P, C)

    @torch.no_grad()
    def score_with_map(self, pil, bank):
        """Anomaly map + scalar score for one image. score = max over patches
        of the patch's nearest-neighbour distance to the memory bank."""
        p = self.patches([pil])[0]                      # (P, C)
        h, w = self.last_hw
        d = torch.cdist(p, bank).min(dim=1).values      # (P,)
        grid = d.reshape(h, w).cpu().numpy()            # (H, W) patch distances
        return grid, float(d.max().item())


def load_imgs(paths):
    out = []
    for p in paths:
        try:
            out.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    return out


def greedy_coreset(feats, k):
    n = feats.shape[0]
    if n <= k:
        return feats
    if n > CAND_CAP:
        idx = torch.randperm(n, device=feats.device)[:CAND_CAP]
        feats = feats[idx]
        n = CAND_CAP
    sel = [int(torch.randint(0, n, (1,)).item())]
    min_d = torch.cdist(feats, feats[sel[0]:sel[0] + 1]).squeeze(1)
    for _ in range(k - 1):
        nxt = int(torch.argmax(min_d).item())
        sel.append(nxt)
        d = torch.cdist(feats, feats[nxt:nxt + 1]).squeeze(1)
        min_d = torch.minimum(min_d, d)
    return feats[sel]


@torch.no_grad()
def image_scores(ext, paths, bank, bs=32):
    scores = []
    for i in range(0, len(paths), bs):
        ims = load_imgs(paths[i:i + bs])
        if not ims:
            continue
        p = ext.patches(ims)                       # (B,P,C)
        b = p.shape[0]
        for j in range(b):
            d = torch.cdist(p[j], bank)            # (P, M)
            patch_min = d.min(dim=1).values        # (P,)
            scores.append(float(patch_min.max().item()))
    return scores


def auroc(neg, pos):
    # neg=normal(label0), pos=anomaly(label1); rank-based AUROC
    y = [0] * len(neg) + [1] * len(pos)
    s = list(neg) + list(pos)
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0] * len(s)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    npos = sum(y)
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    sum_pos = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all SKUs
    os.makedirs(BANK_DIR, exist_ok=True)
    split = json.load(open(SPLIT_JSON, encoding="utf-8"))
    skus = split["skus"]
    if limit:
        skus = skus[:limit]
    ext = Extractor()

    results = {}
    t0 = time.time()
    for n, sku in enumerate(skus):
        sdir = os.path.join(DATASET, sku)
        tr = [os.path.join(sdir, f) for f in split["splits"][sku]["train"]]
        va = [os.path.join(sdir, f) for f in split["splits"][sku]["val"]]
        # fit memory bank
        feats = []
        for i in range(0, len(tr), 32):
            ims = load_imgs(tr[i:i + 32])
            if ims:
                p = ext.patches(ims)
                feats.append(p.reshape(-1, p.shape[-1]))
        bank_all = torch.cat(feats, dim=0)
        bank = greedy_coreset(bank_all, CORESET).contiguous()
        # own val scores
        own = image_scores(ext, va, bank)
        # proxy anomalies: other SKUs' images
        others = [s for s in skus if s != sku]
        random.shuffle(others)
        neg_paths = []
        for os_ in others[:N_NEG_SKUS]:
            od = os.path.join(DATASET, os_)
            ofs = split["splits"][os_]["val"][:N_NEG_PER]
            neg_paths += [os.path.join(od, f) for f in ofs]
        other = image_scores(ext, neg_paths, bank)
        roc = auroc(own, other)
        thr = float(np.percentile(own, 99)) if own else float("nan")
        np.savez(os.path.join(BANK_DIR, sku + ".npz"),
                 bank=bank.half().cpu().numpy(), threshold=thr)
        results[sku] = {
            "n_train": len(tr), "n_val": len(va), "bank_size": int(bank.shape[0]),
            "own_mean": float(np.mean(own)) if own else None,
            "own_std": float(np.std(own)) if own else None,
            "own_max": float(np.max(own)) if own else None,
            "other_mean": float(np.mean(other)) if other else None,
            "threshold_p99": thr,
            "separability_auroc": roc,
        }
        if (n + 1) % 10 == 0 or n == 0:
            print("[%d/%d] %s auroc=%.3f bank=%d" % (n + 1, len(skus), sku, roc, bank.shape[0]), flush=True)

    rocs = [r["separability_auroc"] for r in results.values() if r["separability_auroc"] == r["separability_auroc"]]
    agg = {
        "model": "PatchCore (wide_resnet50_2, layer2+3, greedy coreset)",
        "constraint": "normal-data-only; cross-SKU separability proxy (NOT defect detection)",
        "img_size": IMG, "coreset": CORESET, "num_skus": len(results),
        "total_seconds": round(time.time() - t0, 1),
        "mean_separability_auroc": float(np.mean(rocs)) if rocs else None,
        "median_separability_auroc": float(np.median(rocs)) if rocs else None,
        "min_separability_auroc": float(np.min(rocs)) if rocs else None,
        "per_sku": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print("PatchCore done: %d SKUs, mean separability AUROC=%.4f, %.1fs"
          % (len(results), agg["mean_separability_auroc"] or 0, agg["total_seconds"]))


if __name__ == "__main__":
    main()
