import sys
import cv2
import numpy as np
import os
import re
import pandas as pd
from skimage.morphology import skeletonize

from matplotlib import pyplot as plt

from dicom_spacing import mm_per_pixel, image_size

# 輪郭の隙間を閉じるカーネルサイズ (px)
CLOSE_KERNEL = 7

# 結果を保存するディレクトリ名
OUTPUT_RESULT_DIRECTORY = "biometries_result"
OUTPUT_IMAGE_DIRECTORY = os.path.join(OUTPUT_RESULT_DIRECTORY, "tcd_images")
# 結果を保存するCSVファイル
OUTPUT_CSV_FILE = os.path.join(OUTPUT_RESULT_DIRECTORY, "tcd_measurement.csv")

TARGET_CASES = [""]

def load_mask(path):
    # マスクをグレースケールで読み込み、2値化する
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return (image > 0).astype(np.uint8)


def get_endpoints(axis_mask):
    lines = cv2.HoughLinesP(
        axis_mask * 255,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10
    )
    if lines is None:
        return None

    # HoughLinesP は (N, 1, 4) の配列を返す
    x1, y1, x2, y2 = lines[0][0]
    return (int(x1), int(y1)), (int(x2), int(y2))


def get_unit_vector(p1, p2):
    v = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    return v / np.linalg.norm(v)


def prepare_cerebellum(outline_mask, centerline=True):
    """
    輪郭マスクから
      - points : 弦の端点候補 (x, y)
      - filled : 小脳領域を塗りつぶしたマスク（弦が領域内にあるかの判定用）
    を作る。
    """
    # 輪郭の小さな隙間を閉じる（閉じていないと塗りつぶしに失敗する）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    closed = cv2.morphologyEx(outline_mask, cv2.MORPH_CLOSE, kernel)

    # 外側の輪郭で塗りつぶし
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)

    if centerline:
        # 輪郭線の太さ（約5px）の影響を除くため、線の中心（スケルトン）を使う
        line = skeletonize(closed > 0)
    else:
        line = closed > 0

    ys, xs = np.nonzero(line & (filled > 0))
    points = np.column_stack([xs, ys])
    return points, filled


def chord_inside(filled, a, b):
    # 弦上の点がすべて領域内にあるか
    n = int(np.ceil(np.hypot(*(b - a)))) + 1
    xs = np.round(np.linspace(a[0], b[0], n)).astype(int)
    ys = np.round(np.linspace(a[1], b[1], n)).astype(int)
    return bool(np.all(filled[ys, xs] > 0))


def find_TCD(outline_mask, p1, p2, centerline=True, spacing=(1.0, 1.0)):
    # spacing = (sx, sy): 1ピクセルの大きさ (mm)。既定の (1, 1) ならピクセル単位
    sx, sy = spacing
    v = get_unit_vector(p1, p2)
    points, filled = prepare_cerebellum(outline_mask, centerline)
    if points is None or len(points) < 2:
        return 0.0, None, []

    vecs = points - np.array(p1)
    t = vecs @ v                                  # 主軸方向の座標
    s = v[0] * vecs[:, 1] - v[1] * vecs[:, 0]      # 主軸に垂直な方向の座標

    # 主軸方向に1pxごとにビン分け（各点はちょうど1つのビンに入る）
    bins = np.round(t).astype(int)

    max_length = 0.0
    best_pair = None
    chords = []

    for b in np.unique(bins):
        idx = np.nonzero(bins == b)[0]
        if len(idx) < 2:
            continue
        a_pt = points[idx[np.argmin(s[idx])]]
        b_pt = points[idx[np.argmax(s[idx])]]

        # 弦が領域外を通っていれば無効
        if not chord_inside(filled, a_pt, b_pt):
            continue

        # x と y にそれぞれの間隔を掛けてから長さを計算する
        d = b_pt - a_pt
        length = float(np.hypot(d[0] * sx, d[1] * sy))
        chords.append((a_pt, b_pt))
        if length > max_length:
            max_length = length
            best_pair = (a_pt, b_pt)

    return max_length, best_pair, chords


def tcd_measurement(axis_path, cerebellum_path, dcm_path=None):
    axis = load_mask(axis_path)
    cerebellum = load_mask(cerebellum_path)

    spacing, unit = (1.0, 1.0), "px"
    if dcm_path is not None:
        print("Using DICOM information")
        # マスクがDICOMと同じ画素グリッドでなければ変換できない
        h, w = cerebellum.shape
        if image_size(dcm_path) != (w, h):
            raise ValueError(
                f"Mask size {(w, h)} does not match DICOM size {image_size(dcm_path)}"
            )
        spacing = mm_per_pixel(dcm_path)
        if spacing is None:
            raise ValueError("No pixel spacing found in DICOM")
        unit = "mm"

    endpoints = get_endpoints(axis)
    if endpoints is None:
        raise RuntimeError("No line found in principal axis image")
    a, b = endpoints

    tcd, pair, chords = find_TCD(cerebellum, a, b, spacing=spacing)
    if pair is None:
        raise RuntimeError("No valid chord found")

    # 結果画像
    img = cv2.cvtColor(cerebellum * 255, cv2.COLOR_GRAY2BGR)
    for c1, c2 in chords[::10]:
        cv2.line(img, tuple(map(int, c1)), tuple(map(int, c2)), (120, 120, 120), 1)
    img[axis > 0] = (0, 0, 255)
    cv2.line(img, tuple(map(int, pair[0])), tuple(map(int, pair[1])), (0, 255, 0), 2)
    cv2.imwrite("result_tcd.png", img)

    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.show()

    print(f"TCD: {tcd:.2f} {unit}")

    return img, tcd

def main(argv):
    # argv: principal axis folder path, cerebellum folder path, [DICOM folder path]
    axis_folder = argv[1]
    cerebellum_folder = argv[2]
    dcm_folder = argv[3]

    axis_dict = {}
    cerebellum_dict = {}
    dcm_dict = {}

    # 症例の順番を一致する
    for filename in os.listdir(axis_folder):
        m = re.search(r"result_(.+)_principal_axis", filename)
        if m:
            n = m.group(1)
            axis_dict[n] = os.path.join(axis_folder, filename)

    for filename in os.listdir(cerebellum_folder):
        m = re.search(r"result_(.+)_cerebel", filename)
        if m:
            n = m.group(1)
            cerebellum_dict[n] = os.path.join(cerebellum_folder, filename)

    for filename in os.listdir(dcm_folder):
        m = re.fullmatch(r"(.+)\.dcm", filename)
        if m:
            n = m.group(1)
            dcm_dict[n] = os.path.join(dcm_folder, filename)

    # 主軸・小脳・dcm のすべてが存在する症例を使う
    common_n = sorted(set(axis_dict) & set(cerebellum_dict) & set(dcm_dict))
    print(len(common_n))

    # 結果を保存するフォルダ、辞書を作成
    os.makedirs(OUTPUT_RESULT_DIRECTORY, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_DIRECTORY, exist_ok=True)
    result_dict = {"xcf_name":[], "tcd(mm)":[]}

    # 各症例について、計測する
    for n in common_n:
        axis_path = axis_dict[n]
        cerebellum_path = cerebellum_dict[n]
        dcm_path = dcm_dict[n]

        print(f"Now, case {n} processing...")
        result_img, tcd = tcd_measurement(axis_path, cerebellum_path, dcm_path)

        # 結果の保存！
        result_dict["xcf_name"].append(n)
        result_dict["tcd(mm)"].append(tcd)
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIRECTORY, f"tcd_{n}_img.png"), result_img)

    df = pd.DataFrame(result_dict)
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8-sig")

if __name__ == '__main__':
    main(sys.argv)
