import argparse
import zipfile
from pathlib import Path


PACKAGE_FILES = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "app_paths.py",
    "main.py",
    "pose_compat.py",
    "scripts/build_windows_exe.py",
    "scripts/build_windows_installer.py",
    "scripts/make_release_package.py",
    "installer/RealTimeYoga.iss",
    "test_angle.py",
    "test tts.py",
    "tts.py",
    "angle.txt",
    "audio1.mp3",
    "audio1.wav",
    "easy.png",
    "TREE-M.png",
    "TREE-W.png",
    "TREE-W-orign.png",
    "pose_landmarker_lite.task",
]


def make_package(version: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    dist = root / "dist"
    dist.mkdir(exist_ok=True)

    archive = dist / f"RealTimeYoga-{version}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for relative_path in PACKAGE_FILES:
            source = root / relative_path
            if source.exists():
                zf.write(source, Path(f"RealTimeYoga-{version}") / relative_path)

    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args()

    archive = make_package(args.version)
    print(archive)


if __name__ == "__main__":
    main()
