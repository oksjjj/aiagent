#!/bin/bash
# -------------------------------------
# Node.js + npm + npx 설치 스크립트
# 작성자: Teddy Lee (Braincrew)
# -------------------------------------

set -e  # 에러 발생 시 즉시 중단

# 1. NodeSource에서 Node.js 20.x 설치 스크립트 실행
echo "🔹 NodeSource 저장소 설정 중..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -

# 2. Node.js 및 npm 설치
echo "🔹 Node.js 및 npm 설치 중..."
sudo apt-get install -y nodejs

# 3. npm 최신 버전으로 업데이트
echo "🔹 npm 최신 버전으로 업데이트 중..."
sudo npm install -g npm@latest

# 4. npx 수동 설치/업데이트 (혹시 빠진 경우 대비)
echo "🔹 npx 설치 중..."
sudo npm install -g npx

# 5. 버전 확인
echo ""
echo "✅ 설치 완료!"
echo "-----------------------------"
echo "Node.js 버전: $(node -v)"
echo "npm 버전: $(npm -v)"
echo "npx 버전: $(npx --version)"
echo "-----------------------------"