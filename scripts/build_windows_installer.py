import argparse
import os
import subprocess
from pathlib import Path


APP_NAME = "RealTimeYoga"


def find_iscc() -> Path:
    env_path = os.environ.get("INNO_SETUP_COMPILER")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend([
        Path("ISCC.exe"),
        Path.home() / "AppData/Local/Programs/Inno Setup 6/ISCC.exe",
        Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
        Path("C:/Program Files/Inno Setup 6/ISCC.exe"),
    ])

    for candidate in candidates:
        if candidate.name.lower() == "iscc.exe" and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "ISCC.exe was not found. Install Inno Setup 6 or set INNO_SETUP_COMPILER."
    )


def build_installer(version: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    app_dir = root / "dist" / APP_NAME
    app_exe = app_dir / f"{APP_NAME}.exe"
    if not app_exe.exists():
        raise FileNotFoundError(f"Build the Windows EXE package first: {app_exe}")

    output = root / "dist" / f"{APP_NAME}-Setup-{version}.exe"
    if output.exists():
        output.unlink()

    iscc = find_iscc()
    script = root / "installer" / "RealTimeYoga.iss"
    subprocess.run(
        [
            str(iscc),
            f"/DAppVersion={version}",
            f"/DSourceDir={app_dir}",
            f"/DOutputDir={root / 'dist'}",
            str(script),
        ],
        check=True,
        cwd=root,
    )

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args()

    installer = build_installer(args.version)
    print(installer)


if __name__ == "__main__":
    main()
