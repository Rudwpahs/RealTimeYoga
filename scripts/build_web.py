import argparse
import shutil
import zipfile
from pathlib import Path


WEB_FILES = [
    "index.html",
    "styles.css",
    "app.js",
]

ASSET_FILES = [
    "audio1.wav",
    "easy.png",
    "TREE-M.png",
    "TREE-W.png",
    "pose_landmarker_lite.task",
]


def build_web(version: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    web_root = root / "web"
    dist = root / "dist"
    output = dist / "web"
    assets = output / "assets"

    shutil.rmtree(output, ignore_errors=True)
    assets.mkdir(parents=True, exist_ok=True)

    for relative_path in WEB_FILES:
        shutil.copy2(web_root / relative_path, output / relative_path)

    for relative_path in ASSET_FILES:
        shutil.copy2(root / relative_path, assets / relative_path)

    archive = dist / f"RealTimeYoga-Web-{version}.zip"
    if archive.exists():
        archive.unlink()

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output.rglob("*"):
            if path.is_file():
                zf.write(path, Path(f"RealTimeYoga-Web-{version}") / path.relative_to(output))

    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args()

    archive = build_web(args.version)
    print(archive)


if __name__ == "__main__":
    main()
