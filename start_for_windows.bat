@echo off
chcp 65001 >nul
cd /d "%~dp0"

title Pill Detection Launcher

echo ========================================
echo    Pill Detection 자동 실행
echo ========================================
echo.

echo [STEP 1] 환경 설치 중...
python serve\setup_serve.py

if %errorlevel% neq 0 (
    echo.
    echo [FAIL] 설치 실패
    pause
    exit /b
)

echo.
echo [STEP 1 완료]
echo.

set /p answer=서버를 실행하시겠습니까? (Y/N): 

if /i "%answer%"=="Y" (
    echo.
    echo [STEP 2] 서버 실행 중...
    echo.
    python serve\run_server.py
) else (
    echo.
    echo 프로그램을 종료합니다.
)

pause