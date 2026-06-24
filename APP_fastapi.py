from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from patchcore import Extractor

from PIL import Image

import cv2
import numpy as np
import torch

import shutil
import uuid
import os

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
    "uploads_temp",
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

    _, score = extractor.score_with_map(
        img,
        bank
    )

    return score

# =====================================================
# API
# =====================================================

@app.post("/predict")

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
        "uploads_temp",
        filename
    )

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

        cls_id = int(
            boxes.cls[best_idx]
        )

        confidence = float(
            boxes.conf[best_idx]
        )

        drug_name = (
            model.names[cls_id]
        )

        # -------------------------
        # PatchCore npz
        # -------------------------

        npz_path = os.path.join(
            r"C:\Users\human3_12\Desktop\project1\Models\patchcore_200",
            f"{drug_name}.npz"
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

        score = patchcore_predict(
            save_path,
            bank
        )

        is_defect = (
            score >= threshold
        )

        # -------------------------
        # Firebase Storage 업로드
        # -------------------------
        print("Firebase upload start")
        print(save_path)

        blob = bucket.blob(
            f"uploads/{filename}"
        )

        blob.upload_from_filename(
            save_path
        )

        image_url = (
            f"gs://pill-anomaly-detector.firebasestorage.app/uploads/{filename}"
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

            "image_url": image_url
        }

    finally:

        if os.path.exists(save_path):
            os.remove(save_path)

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
        )
    }

# =====================================================
# 실행
# =====================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )