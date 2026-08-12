#!/bin/bash
# Nature2Music 本地 Web 应用启动脚本
# 用法: bash webapp/启动.sh   然后浏览器打开 http://127.0.0.1:8321
set -e
cd "$(dirname "$0")/.."
echo "Nature2Music 启动中… http://127.0.0.1:8321"
exec .venv/bin/python webapp/server.py
