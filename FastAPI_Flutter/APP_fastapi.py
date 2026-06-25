from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from datetime import timedelta
from patchcore import Extractor

from PIL import Image

import cv2
import numpy as np
import torch
import base64
import shutil
import uuid
import os
import uvicorn
import json

import firebase_admin

from firebase_admin import (
    credentials,
    storage,
    firestore
)

# =====================================================
# FastAPI
# =====================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cred = credentials.Certificate(
    "firebase-key.json"
)

firebase_admin.initialize_app(
    cred,
    {
        "storageBucket":
        "pill-anomaly-detector.firebasestorage.app"
    }
)

db = firestore.client()

bucket = storage.bucket()

# =====================================================
# 핸드폰 사진 흔들림, 초점나감 등 카메라 이상 감지
# =====================================================

def is_blurry(image_path, threshold=4):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return score < threshold, score

# =====================================================
# YOLO
# =====================================================

model = YOLO(
    "Models/YOLO_best.pt"
)

# =====================================================
# 약품명 매핑 (SKU 코드 -> 사람이 읽는 약품명)
# 파일 없으면 빈 dict -> 코드 그대로 노출(폴백)
# =====================================================

_NAMES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "drug_names.json"
)

try:
    with open(_NAMES_PATH, encoding="utf-8") as _f:
        DRUG_NAMES = json.load(_f)
except (OSError, ValueError):
    DRUG_NAMES = {}

# =====================================================
# PatchCore
# =====================================================

extractor = Extractor()

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# Upload 폴더
# =====================================================

os.makedirs(
    "uploads",
    exist_ok=True
)

# =====================================================
# PatchCore 추론
# =====================================================

def patchcore_predict(
    image_path,
    bank
):

    img = Image.open(
        image_path
    ).convert("RGB")

    heatmap, score = extractor.score_with_map(
        img,
        bank
    )

    print(type(heatmap))

    try:
        print(heatmap.shape)
    except:
        print("shape 없음")

    return heatmap, score


# =====================================================
# Heatmap 추가
# =====================================================

def make_heatmap_overlay(
    heatmap,
    image_path,
    alpha=0.45
):

    base = cv2.imread(image_path)

    h, w = base.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = cv2.normalize(
        heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    threshold = 180  # 0~255

    mask = heatmap > threshold

    overlay = base.copy()

    overlay[mask] = (
        overlay[mask] * 0.4
        + np.array([0, 0, 255]) * 0.6
    ).astype(np.uint8)


    ok, buf = cv2.imencode(
        ".jpg",
        overlay
    )

    if not ok:
        return None

    return base64.b64encode(
        buf
    ).decode("utf-8")

# =====================================================
# API
# =====================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -------------------------
    # 파일 저장
    # -------------------------

    filename = (
        f"{uuid.uuid4()}.png"
    )

    save_path = os.path.join(
        "uploads",
        filename
    )

    crop_path = None

    try:

        with open(
            save_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # -------------------------
        # 카메라 이상 탐지
        # -------------------------

        is_blur, blur_score = is_blurry(
            save_path
        )

        print("Blur Score =", blur_score)

        if is_blur:
            print("RETURN")
            
            return {

                "detected": False,

                "is_blur": True,

                "blur_score": round(
                    blur_score,
                    2
                ),

                "message":
                "사진이 흐립니다. 카메라 렌즈를 닦고 다시 촬영해주세요."
            }

        # -------------------------
        # YOLO
        # -------------------------

        results = model(
            save_path,
            verbose=False
        )

        boxes = results[0].boxes

        if len(boxes) == 0:

            return {
                "detected": False,
                "message": "약품 검출 실패"
            }

        best_idx = boxes.conf.argmax()

        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        img = cv2.imread(save_path)
        crop = img[y1:y2, x1:x2]

        crop_path = os.path.join(
            "uploads",
            f"crop_{filename}"
        )

        cv2.imwrite(
            crop_path,
            crop
        )

        

        cls_id = int(
            boxes.cls[best_idx]
        )

        confidence = float(
            boxes.conf[best_idx]
        )

        drug_id = (
            model.names[cls_id]
        )

        # SKU 코드 -> 약품명 (매핑 없으면 코드 그대로)
        drug_name = DRUG_NAMES.get(
            drug_id,
            drug_id
        )

        # -------------------------
        # PatchCore npz  (파일명은 SKU 코드)
        # -------------------------

        npz_path = os.path.join(
            r"C:\Users\human3_12\Desktop\project1\Models\patchcore_200",
            f"{drug_id}.npz"
        )

        if not os.path.exists(
            npz_path
        ):

            return {
                "detected": True,
                "drug_name": drug_name,
                "confidence": round(
                    confidence,
                    4
                ),
                "patchcore_loaded": False
            }

        data = np.load(
            npz_path,
            allow_pickle=True
        )

        threshold = float(
            data["threshold"]
        )

        bank = torch.tensor(
            data["bank"],
            dtype=torch.float32,
            device=DEVICE
        )

        # -------------------------
        # PatchCore 추론
        # -------------------------

        heatmap, score = patchcore_predict(
            crop_path,
            bank
        )
        

        is_defect = (
            score >= threshold
        )

        if is_defect:
            
            heatmap_image = make_heatmap_overlay(
            heatmap,
            crop_path
        )

        else:
            
            heatmap_image = None


        # -------------------------
        # Firebase Storage 업로드
        # -------------------------
        print("Firebase upload start")
        print(save_path)

        image_blob = bucket.blob(
            f"uploads/{filename}"
        )

        image_blob.upload_from_filename(
            save_path
        )

        image_url = image_blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=7),
            method="GET"
        )

        print("Firebase upload success")

        # -------------------------
        # 결과
        # -------------------------

        return {

            "detected": True,

            "drug_name": drug_name,

            "confidence": round(
                confidence,
                4
            ),

            "anomaly_score": round(
                score,
                4
            ),

            "threshold": round(
                threshold,
                4
            ),

            "is_defect": bool(
                is_defect
            ),

            "patchcore_loaded": True,

            "bank_size": int(
                bank.shape[0]
            ),

            "image_url": image_url,
            "heatmap_image": heatmap_image
        }

    finally:

        for p in [
            save_path,
            crop_path
        ]:

            if p and os.path.exists(p):
                os.remove(p)
# =====================================================
# 실행
# =====================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
