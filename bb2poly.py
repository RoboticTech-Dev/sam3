import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import SAM


IMG_EXTS = {".JPG", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> List[float]:
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h

    # clip
    x1 = float(max(0, min(img_w - 1, x1)))
    y1 = float(max(0, min(img_h - 1, y1)))
    x2 = float(max(0, min(img_w - 1, x2)))
    y2 = float(max(0, min(img_h - 1, y2)))
    return [x1, y1, x2, y2]


def read_yolo_bboxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        rows.append((cls, cx, cy, w, h))
    return rows


def polygon_to_yolo_seg_line(
    cls: int, poly_xy: np.ndarray, img_w: int, img_h: int
) -> str:
    # poly_xy: (N,2) in pixels -> normalize to [0,1]
    xs = poly_xy[:, 0] / float(img_w)
    ys = poly_xy[:, 1] / float(img_h)
    coords = []
    for x, y in zip(xs, ys):
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    return f"{cls} " + " ".join(coords)


def main(
    images_dir: str, labels_dir: str, out_labels_dir: str, model_path: str, imgsz: int
):
    images_dir_p = Path(images_dir)
    labels_dir_p = Path(labels_dir)
    out_labels_dir_p = Path(out_labels_dir)
    out_labels_dir_p.mkdir(parents=True, exist_ok=True)

    model = SAM(model_path)

    image_paths = [
        p
        for p in images_dir_p.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    image_paths.sort()

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        label_path = labels_dir_p / f"{img_path.stem}.txt"
        yolo_rows = read_yolo_bboxes(label_path)
        if not yolo_rows:
            # se quiser gerar vazio, descomente:
            # (out_labels_dir_p / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
            continue

        classes = []
        bboxes_xyxy = []
        for cls, cx, cy, w, h in yolo_rows:
            classes.append(cls)
            bboxes_xyxy.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))

        # SAM3 visual prompt por bbox (segmenta dentro da bbox)
        # API de bbox prompt: model.predict(..., bboxes=[x1,y1,x2,y2]) :contentReference[oaicite:2]{index=2}
        results = model.predict(
            source=str(img_path), bboxes=bboxes_xyxy, imgsz=imgsz, verbose=False
        )

        if not results or results[0].masks is None:
            continue

        # Ultralytics Results: masks.xy -> lista de polígonos (em pixels) por instância
        polys = results[0].masks.xy  # List[np.ndarray], cada um (Ni,2)

        out_lines = []
        n = min(len(polys), len(classes))
        for i in range(n):
            poly = polys[i]
            if poly is None or len(poly) < 3:
                continue
            poly = np.asarray(poly, dtype=np.float32)

            # garante dentro da imagem
            poly[:, 0] = np.clip(poly[:, 0], 0, img_w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, img_h - 1)

            out_lines.append(polygon_to_yolo_seg_line(classes[i], poly, img_w, img_h))

        (out_labels_dir_p / f"{img_path.stem}.txt").write_text(
            "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", required=True, help="Pasta com as imagens")
    parser.add_argument(
        "-l", "--labels", required=True, help="Pasta com labels YOLO bbox (.txt)"
    )
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="Pasta de saída para labels YOLO-seg (polígonos)",
    )
    parser.add_argument(
        "-m", "--model", default="sam3.pt", help="Caminho do modelo SAM3 (ex: sam3.pt)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamanho de inferência",
    )
    args = parser.parse_args()

    main(args.images, args.labels, args.out, args.model, args.imgsz)
