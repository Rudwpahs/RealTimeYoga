import argparse
import shutil
import zipfile
from pathlib import Path

import PyInstaller.__main__


APP_NAME = "RealTimeYoga"

DATA_FILES = [
    "README.md",
    "requirements.txt",
    "angle.txt",
    "audio1.mp3",
    "audio1.wav",
    "easy.png",
    "TREE-M.png",
    "TREE-W.png",
    "TREE-W-orign.png",
    "pose_landmarker_lite.task",
]


def build_exe(version: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    dist = root / "dist"
    app_dist = dist / APP_NAME

    shutil.rmtree(root / "build", ignore_errors=True)
    shutil.rmtree(app_dist, ignore_errors=True)

    args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--specpath",
        str(root / "build"),
        "--name",
        APP_NAME,
        "--collect-all",
        "mediapipe",
        "--exclude-module",
        "mediapipe.tasks.python.benchmark",
        "--exclude-module",
        "mediapipe.tasks.python.genai",
        "--exclude-module",
        "mediapipe.tasks.python.metadata",
        "--exclude-module",
        "mediapipe.tasks.python.test",
        "--exclude-module",
        "pandas",
        "--exclude-module",
        "pyarrow",
        "--exclude-module",
        "scipy",
        "--exclude-module",
        "torch",
        "--exclude-module",
        "torchvision",
    ]

    for relative_path in DATA_FILES:
        source = root / relative_path
        if source.exists():
            args.extend(["--add-data", f"{source}{';.'}"])

    args.append(str(root / "main.py"))
    PyInstaller.__main__.run(args)

    archive = dist / f"{APP_NAME}-Windows-{version}.zip"
    if archive.exists():
        archive.unlink()

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in app_dist.rglob("*"):
            if path.is_file():
                zf.write(path, Path(f"{APP_NAME}-Windows-{version}") / path.relative_to(app_dist))

    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args()

    archive = build_exe(args.version)
    print(archive)


if __name__ == "__main__":
    main()
