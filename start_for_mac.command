#!/bin/bash

cd "$(dirname "$0")"

clear
echo "========================================"
echo "   Pill Detection 자동 실행"
echo "========================================"
echo

echo "[STEP 1] 환경 설치 시작..."
python3 serve/setup_serve.py

if [ $? -ne 0 ]; then
  echo
  echo "[FAIL] setup 실패"
  read -n 1 -s -r -p "아무 키나 누르면 종료합니다..."
  echo
  exit 1
fi

echo
read -p "서버를 실행하시겠습니까? (Y/N): " answer

if [[ "$answer" == "Y" || "$answer" == "y" ]]; then
  echo
  echo "[STEP 2] 서버 실행 중..."
  python3 serve/run_server.py
else
  echo
  echo "실행을 취소했습니다."
fi

echo
read -n 1 -s -r -p "아무 키나 누르면 종료합니다..."
echo