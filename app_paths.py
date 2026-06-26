import sys
from pathlib import Path


def resource_path(relative_path: str) -> str:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str(base_path / relative_path)


def read_image(path: str):
    import cv2
    import numpy as np

    candidate_paths = [Path(path)]
    if not candidate_paths[0].is_absolute():
        candidate_paths.append(Path(resource_path(path)))

    for candidate_path in candidate_paths:
        try:
            data = np.fromfile(str(candidate_path), dtype=np.uint8)
        except OSError:
            continue

        if data.size == 0:
            continue

        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img

    return None
