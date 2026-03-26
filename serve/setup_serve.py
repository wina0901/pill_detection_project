from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / "venv"


def info(msg): print(f"[INFO] {msg}")
def ok(msg): print(f"[OK]   {msg}")
def err(msg): print(f"[ERROR] {msg}")


def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        raise RuntimeError("명령 실행 실패")


def get_python():
    return VENV_DIR / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def get_pip():
    return VENV_DIR / ("Scripts/pip.exe" if platform.system() == "Windows" else "bin/pip")


def recreate_venv():
    if VENV_DIR.exists():
        info("기존 venv 삭제 중...")
        shutil.rmtree(VENV_DIR)
        ok("venv 삭제 완료")

    info("새 venv 생성 중...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    ok("venv 생성 완료")


def install():
    py = get_python()
    pip = get_pip()

    info(f"Python: {py}")

    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    info("PyTorch 설치...")
    run([str(pip), "install", "torch", "torchvision", "torchaudio"])

    info("requirements 설치...")
    run([str(pip), "install", "-r", str(PROJECT_ROOT / "requirements-serve.txt")])

    # 🔥 핵심: qrcode 강제 설치 (이게 포인트)
    info("qrcode 강제 설치 확인...")
    run([str(pip), "install", "qrcode"])

    ok("패키지 설치 완료")


def ensure_dirs():
    for d in ["temp", "results", "results/crops"]:
        p = PROJECT_ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        print(f"  - {p}")


def main():
    try:
        print("="*50)
        print("SETUP START")
        print("="*50)

        clean = "--clean" in sys.argv

        if clean:
            recreate_venv()
        elif not VENV_DIR.exists():
            recreate_venv()
        else:
            info("기존 venv 사용")

        install()
        ensure_dirs()

        print("\n" + "="*50)
        ok("SETUP 완료")
        print("실행: python serve/run_server.py")
        print("="*50)

    except Exception as e:
        print("\n" + "="*50)
        err(str(e))
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    main()