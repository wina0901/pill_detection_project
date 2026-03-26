from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / "venv"


def info(message: str) -> None:
    print(f"[INFO] {message}")


def success(message: str) -> None:
    print(f"[OK]   {message}")


def warning(message: str) -> None:
    print(f"[WARN] {message}")


def error(message: str) -> None:
    print(f"[ERROR] {message}")


def get_venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv_exists() -> Path:
    venv_python = get_venv_python()
    if not venv_python.exists():
        raise FileNotFoundError(
            "가상환경이 없습니다. 먼저 'python serve/setup_serve.py' 를 실행해주세요."
        )
    return venv_python


def reexec_into_venv_if_needed() -> None:
    """
    현재 실행 중인 Python이 venv Python이 아니면,
    venv Python으로 이 스크립트를 다시 실행한다.
    """
    venv_python = ensure_venv_exists()

    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()

    if current_python == target_python:
        return

    info("현재 Python이 venv가 아니어서 venv Python으로 다시 실행합니다.")
    info(f"현재 Python: {current_python}")
    info(f"venv Python: {target_python}")

    cmd = [str(target_python), str(Path(__file__).resolve())] + sys.argv[1:]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


def ensure_dirs() -> None:
    for folder in ["temp", "results", "results/crops"]:
        (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)


def check_required_files() -> None:
    required_files = [
        PROJECT_ROOT / "server.py",
        PROJECT_ROOT / "requirements-serve.txt",
        PROJECT_ROOT / "data" / "meta.csv",
        PROJECT_ROOT / "data" / "merged_annotations_train_final.json",
        PROJECT_ROOT / "ui" / "templates" / "index.html",
        PROJECT_ROOT / "ui" / "static" / "style.css",
    ]

    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "필수 파일이 없습니다:\n- " + "\n- ".join(missing)
        )


def wait_for_server(host: str, port: int, timeout: int = 20) -> bool:
    start = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            if time.time() - start > timeout:
                return False
            time.sleep(0.3)


def get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        sock.close()
    return ip


def print_qr_to_terminal(url: str) -> None:
    """
    qrcode는 venv Python으로 재실행된 뒤에 import 가능
    """
    import qrcode

    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    matrix = qr.get_matrix()

    print("\n[INFO] 모바일 접속용 QR 코드\n")
    print("  " + "██" * (len(matrix[0]) + 2))
    for row in matrix:
        line = "  ██"
        for cell in row:
            line += "  " if cell else "██"
        line += "██"
        print(line)
    print("  " + "██" * (len(matrix[0]) + 2))
    print()


def save_qr_image(url: str) -> Path:
    import qrcode

    qr_path = PROJECT_ROOT / "results" / "server_qr.png"
    img = qrcode.make(url)
    img.save(qr_path)
    return qr_path


def main() -> None:
    try:
        # 가장 먼저 venv Python으로 재실행
        reexec_into_venv_if_needed()

        print("=" * 60)
        print("Pill Detection Server Runner")
        print("=" * 60)

        check_required_files()
        ensure_dirs()

        venv_python = Path(sys.executable).resolve()

        host = "0.0.0.0"
        port = 8000

        local_ip = get_local_ip()
        local_url = f"http://127.0.0.1:{port}"
        mobile_url = f"http://{local_ip}:{port}"

        info(f"운영체제: {platform.system()}")
        info(f"프로젝트 경로: {PROJECT_ROOT}")
        info(f"사용 Python: {venv_python}")
        info(f"PC 접속 주소: {local_url}")
        info(f"모바일 접속 주소: {mobile_url}")
        info("서버를 시작합니다...")

        cmd = [
            str(venv_python),
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            host,
            "--port",
            str(port),
            "--reload",
        ]

        process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))

        if wait_for_server("127.0.0.1", port):
            success("서버 준비 완료")
            try:
                webbrowser.open(local_url)
                success("브라우저 실행 완료")
            except Exception:
                warning("브라우저 자동 실행에 실패했습니다.")

            print_qr_to_terminal(mobile_url)
            qr_path = save_qr_image(mobile_url)
            info(f"QR 이미지 저장 위치: {qr_path}")
            info("휴대폰이 같은 와이파이에 연결되어 있어야 접속됩니다.")
        else:
            warning("서버 응답 확인 실패. 브라우저/QR 출력은 생략합니다.")

        process.wait()

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        info("서버를 종료합니다.")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        error(str(e))
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()