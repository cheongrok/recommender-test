#!/bin/bash

# 기본값 설정
EXP_NAME=""
PLACE_NAME=""

# 인자 처리
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--exp-name) EXP_NAME="$2"; shift ;;   # EXP_NAME 지정
        -p|--place-name) PLACE_NAME="$2"; shift ;; # PLACE_NAME 지정
        *) echo "Unknown parameter passed: $1"; exit 1 ;; # 잘못된 인자 처리
    esac
    shift
done

# 체크: 필수 인자가 설정되었는지 확인
if [[ -z "$EXP_NAME" ]] || [[ -z "$PLACE_NAME" ]]; then
    echo "Usage: $0 -e EXP_NAME -p PLACE_NAME"
    exit 1
fi

# 환경 변수 설정
export EXP_NAME
export PLACE_NAME

# Docker Compose 실행
docker compose up -d train
