import os
import pandas as pd
import cv2
from pathlib import Path
import ast
import numpy as np
import random

CSV_PATH = r"selected_200_image_list_with_bbox.csv"

df = pd.read_csv(CSV_PATH)

drug_list = df["drug_N"].unique()

print(len(drug_list))
print(drug_list[:10])



def make_crack_from_png(img, crack_png_path):

    print("균열 생성")
    crack = cv2.imread(
        crack_png_path,
        cv2.IMREAD_UNCHANGED
    )

    h, w = img.shape[:2]

    # -------------------
    # 알약 대비 크기
    # -------------------

    scale = random.uniform(
        0.50,
        0.90
    )

    new_w = int(w * scale)
    new_h = int(crack.shape[0]* (new_w / crack.shape[1]))

    new_w = min(new_w, w - 5)
    new_h = min(new_h, h - 5)

    crack = cv2.resize(
        crack,
        (new_w, new_h)
    )

    # -------------------
    # 회전
    # -------------------

    angle = random.randint(
        0,
        360
    )

    M = cv2.getRotationMatrix2D(
        (new_w//2, new_h//2),
        angle,
        1
    )

    crack = cv2.warpAffine(
        crack,
        M,
        (new_w, new_h),
        borderValue=(0,0,0,0)
    )

    # -------------------
    # 중앙 근처
    # -------------------

    x = random.randint(
        int(w*0.2),
        int(w*0.7)
    )

    y = random.randint(
        int(h*0.2),
        int(h*0.7)
    )

    x = min(
        x,
        w-new_w
    )

    y = min(
        y,
        h-new_h
    )

    result = img.copy()

    alpha = crack[:,:,3] / 255.0

    for c in range(3):

        result[
            y:y+new_h,
            x:x+new_w,
            c
        ] = (
            alpha *
            crack[:,:,c]
            +
            (1-alpha) *
            result[
                y:y+new_h,
                x:x+new_w,
                c
            ]
        )

    return result

def make_dent(img):
    print("찍힘 생성")
    out = img.astype(np.float32)

    h, w = out.shape[:2]

    # ------------------
    # 위치
    # ------------------

    cx = random.randint(
        int(w * 0.35),
        int(w * 0.65)
    )

    cy = random.randint(
        int(h * 0.35),
        int(h * 0.65)
    )

    radius = random.randint(
        12,
        22
    )

    # ------------------
    # 불규칙 압흔 생성
    # ------------------

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    pts = []

    n_pts = random.randint(
        5,
        8
    )

    for i in range(n_pts):

        angle = (
            2 * np.pi * i / n_pts
        )

        rr = radius * random.uniform(
            0.4,
            1.0
        )

        x = int(
            cx + rr * np.cos(angle)
        )

        y = int(
            cy + rr * np.sin(angle)
        )

        pts.append([x, y])

    pts = np.array(
        pts,
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [pts],
        255
    )

    # 약간만 블러
    mask = cv2.GaussianBlur(
        mask,
        (9, 9),
        0
    )

    alpha = mask.astype(
        np.float32
    ) / 255.0

    # ------------------
    # 내부 깊게 어둡게
    # ------------------

    center_dark = (
        alpha *
        random.uniform(
            60,
            100
        )
    )

    for c in range(3):

        out[:, :, c] -= center_dark


    # ------------------
    # 살짝 입체감
    # ------------------

    gx = cv2.Sobel(
        alpha,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        alpha,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    light_x = -0.8
    light_y = -0.8

    shade = (
        gx * light_x
        + gy * light_y
    )

    for c in range(3):

        out[:, :, c] += (
            shade *
            random.uniform(
                20,
                40
            )
        )

    out = np.clip(
        out,
        0,
        255
    )

    return out.astype(
        np.uint8
    )

def make_discoloration(img):
    print("변색 생성")
    out = img.copy()

    h, w = img.shape[:2]

    hsv = cv2.cvtColor(
        out,
        cv2.COLOR_BGR2HSV
    )

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    # 중심
    cx = random.randint(
        int(w * 0.3),
        int(w * 0.7)
    )

    cy = random.randint(
        int(h * 0.3),
        int(h * 0.7)
    )

    # 불규칙 다각형
    pts = []

    r = random.randint(
        10,
        30
    )

    for a in np.linspace(
        0,
        2 * np.pi,
        10,
        endpoint=False
    ):

        rr = r * random.uniform(
            0.6,
            1.4
        )

        x = int(
            cx + rr * np.cos(a)
        )

        y = int(
            cy + rr * np.sin(a)
        )

        pts.append([x, y])

    pts = np.array(
        pts,
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [pts],
        255
    )

    # 가장자리 부드럽게
    mask = cv2.GaussianBlur(
        mask,
        (31, 31),
        0
    )

    alpha = mask / 255.0

    # 변색 종류 랜덤
    mode = random.choices(
        population=[
            "yellow",
            "dark",
            "fade",
            "brown",
            "white",
            "bright"
        ],
        weights=[
            35,
            25,
            15,
            15,
            5,
            5
        ]
    )[0]

    if mode == "yellow":

        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] + alpha * 50,
            0,
            255
        )

    elif mode == "dark":

        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] - alpha * 50,
            0,
            255
        )

    elif mode == "fade":

        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] - alpha * 60,
            0,
            255
        )

    elif mode == "brown":

        hsv[:, :, 0] = np.clip(
            hsv[:, :, 0] + alpha * 8,
            0,
            179
        )

        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] + alpha * 30,
            0,
            255
        )

        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] - alpha * 25,
            0,
            255
        )

    elif mode == "white":

        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] - alpha * 90,
            0,
            255
        )

        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] + alpha * 40,
            0,
            255
        )

    elif mode == "bright":

        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] + alpha * 50,
            0,
            255
        )

    out = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    return out

def overlay_png(background, overlay, x, y):
    print("이물질 생성")
    h, w = overlay.shape[:2]

    if x + w > background.shape[1]:
        w = background.shape[1] - x

    if y + h > background.shape[0]:
        h = background.shape[0] - y

    if w <= 0 or h <= 0:
        return background

    overlay = overlay[:h, :w]

    alpha = overlay[:, :, 3] / 255.0

    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c]
            + (1 - alpha) * background[y:y+h, x:x+w, c]
        )

    return background


def make_contamination(
    img,
    hair_png_path="hair.png"
):

    out = img.copy()

    h, w = out.shape[:2]

    # ------------------
    # 이물질 타입 선택
    # ------------------

    mode = random.choices(
        population=[
            "dots",
            "hair",
            "mixed"
        ],
        weights=[
            60,
            20,
            20
        ]
    )[0]

    # ------------------
    # 검은 점
    # ------------------

    if mode in ["dots", "mixed"]:

        for _ in range(
            random.randint(3, 8)
        ):

            x = random.randint(
                int(w * 0.2),
                int(w * 0.8)
            )

            y = random.randint(
                int(h * 0.2),
                int(h * 0.8)
            )

            radius = random.randint(
                2,
                5
            )

            color = random.randint(
                0,
                40
            )

            cv2.circle(
                out,
                (x, y),
                radius,
                (color, color, color),
                -1
            )

    # ------------------
    # 머리카락
    # ------------------

    if mode in ["hair", "mixed"]:

        hair = cv2.imread(
            hair_png_path,
            cv2.IMREAD_UNCHANGED
        )

        if hair is not None:

            scale = random.uniform(
                0.3,
                0.8
            )

            hair = cv2.resize(
                hair,
                None,
                fx=scale,
                fy=scale
            )

            angle = random.uniform(
                0,
                360
            )

            center = (
                hair.shape[1] // 2,
                hair.shape[0] // 2
            )

            M = cv2.getRotationMatrix2D(
                center,
                angle,
                1.0
            )

            hair = cv2.warpAffine(
                hair,
                M,
                (hair.shape[1], hair.shape[0]),
                borderValue=(0,0,0,0)
            )

            x = random.randint(
                0,
                max(
                    0,
                    w - hair.shape[1]
                )
            )

            y = random.randint(
                0,
                max(
                    0,
                    h - hair.shape[0]
                )
            )

            out = overlay_png(
                out,
                hair,
                x,
                y
            )

    return out

def make_chip(img, bbox):
    print("깨짐 생성")
    out = img.copy()

    x, y, bw, bh = bbox

    # 배경색
    bg_color = tuple(
        int(v)
        for v in img[0:30, 0:30].mean(axis=(0,1))
    )

    # 제거 비율
    ratio = random.uniform(
        0.25,
        0.45
    )

    side = random.choice([
        "left",
        "right",
        "top",
        "bottom"
    ])

    # -------------------
    # 좌우 깨짐
    # -------------------

    if side in ["left", "right"]:

        cut_w = int(
            bw * ratio
        )

        if side == "left":

            cv2.rectangle(
                out,
                (x, y),
                (x + cut_w, y + bh),
                bg_color,
                -1
            )

        else:

            cv2.rectangle(
                out,
                (x + bw - cut_w, y),
                (x + bw, y + bh),
                bg_color,
                -1
            )

    # -------------------
    # 상하 깨짐
    # -------------------

    else:

        cut_h = int(
            bh * ratio
        )

        if side == "top":

            cv2.rectangle(
                out,
                (x, y),
                (x + bw, y + cut_h),
                bg_color,
                -1
            )

        else:

            cv2.rectangle(
                out,
                (x, y + bh - cut_h),
                (x + bw, y + bh),
                bg_color,
                -1
            )

    return out

def make_blur(img):
    print("전체 블러링(카메라초점불량, 흔들림) 생성")
    k = random.choice(
        [9, 13, 21, 25, 29]
    )

    out = cv2.GaussianBlur(
        img,
        (k, k),
        0
    )

    return out

# --------------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------------

NORMAL_ROOT = Path(
    r"E:\Normal"
)

CRACK_PNG = Path(r"crack.png")


OUTPUT_ROOT = Path(
    r"E:\Abnormal"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

defect_funcs = {
    "crack": make_crack_from_png,
    "dent": make_dent,
    "discoloration": make_discoloration,
    "contamination": make_contamination,
    "chip": make_chip,
    "blur": make_blur

}

df = pd.read_csv(CSV_PATH)
drug_list = df["drug_N"].unique()

# --------------------------------------------------------------------------------
# 실행 및 총 개수 확인
# --------------------------------------------------------------------------------

for DRUG_N in drug_list:

    print("=" * 50)
    print("처리중:", DRUG_N)

    target = df[
        df["drug_N"] == DRUG_N
        ]

    front_df = (
        target[
            target["drug_dir"] == "앞면"
            ]
        .sample(
            n=min(
                2,
                len(
                    target[
                        target["drug_dir"] == "앞면"
                        ]
                )
            ),
            random_state=42
        )
    )

    back_df = (
        target[
            target["drug_dir"] == "뒷면"
            ]
        .sample(
            n=min(
                2,
                len(
                    target[
                        target["drug_dir"] == "뒷면"
                        ]
                )
            ),
            random_state=42
        )
    )

    selected = pd.concat(
        [front_df, back_df]
    )

    print(
        "selected:",
        DRUG_N,
        len(selected)
    )

    for defect_name, defect_func in defect_funcs.items():

        defect_dir = (OUTPUT_ROOT)

        defect_dir.mkdir(
            exist_ok=True
        )

        for _, row in selected.iterrows():

            img_path = (
                    NORMAL_ROOT /
                    DRUG_N /
                    row["file_name"]
            )

            img = cv2.imread(
                str(img_path)
            )

            if img is None:
                print(
                    "읽기 실패:",
                    img_path
                )

                continue

            # --------------------
            # bbox 읽기
            # --------------------

            x, y, bw, bh = ast.literal_eval(
                row["bbox"]
            )

            # bbox 범위 잘라내기
            pill = img[
                y:y + bh,
                x:x + bw
            ].copy()

            # bbox가 이상한 경우 방지
            if pill.size == 0:
                print(
                    "bbox 오류:",
                    row["file_name"]
                )

                continue

            # --------------------
            # 불량 생성
            # --------------------

            if defect_name == "crack":

                pill_defect = defect_func(
                    pill,
                    CRACK_PNG
                )

                out = img.copy()

                out[
                    y:y + bh,
                    x:x + bw
                ] = pill_defect

            elif defect_name == "chip":
                out = defect_func(
                    img,
                    (x, y, bw, bh)
                )

            elif defect_name == "blur":
                out = defect_func(img)

            else:

                pill_defect = defect_func(
                    pill
                )

                out = img.copy()

                out[
                    y:y + bh,
                    x:x + bw
                ] = pill_defect

            print(defect_name, row["file_name"])

            # --------------------
            # 저장
            # --------------------

            save_name = (
                    Path(
                        row["file_name"]
                    ).stem
                    +
                    f"_{defect_name}.png"
            )

            cv2.imwrite(
                str(
                    defect_dir /
                    save_name
                ),
                out
            )

            print(
                "저장:",
                save_name
            )

    print("완료")

from pathlib import Path

abnormal_root = Path(r"E:\Abnormal")

files = list(
    abnormal_root.glob("*.png")
)

print("불량 이미지 수:", len(files))