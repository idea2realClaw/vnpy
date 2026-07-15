#!/usr/bin/env bash
set -euo pipefail

python=${1:-python3}

echo "=========================================================="
echo " TA-Lib C 库安装脚本 (Linux / macOS)"
echo " 仓库: idea2realClaw/vnpy"
echo "=========================================================="
echo

# 0) 升级 pip / wheel
echo "[0/3] 升级 pip / wheel ..."
"$python" -m pip install --upgrade pip wheel

# 检测系统是否已存在 ta-lib C 库
ta_lib_exists() {
    command -v ta-lib-config >/dev/null 2>&1 || \
    [ -f /usr/local/lib/libta_lib.a ] || \
    [ -f /usr/local/lib/libta_lib.so ] || \
    [ -f /usr/lib/libta_lib.so ] || \
    [ -f /usr/local/lib/libta_lib.dylib ] || \
    [ -f /opt/homebrew/lib/libta_lib.dylib ]
}

install_ta_lib() {
    uname_s=$(uname -s)
    if [ "$uname_s" = "Darwin" ]; then
        echo "[2/3] macOS: 通过 Homebrew 安装 ta-lib C 库 ..."
        brew install ta-lib
    else
        echo "[2/3] Linux: 从源码编译安装 ta-lib C 库 ..."
        tmp=$(mktemp -d)
        cd "$tmp"
        url="https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz"
        if command -v wget >/dev/null 2>&1; then
            wget -O ta-lib.tar.gz "$url"
        else
            curl -L -o ta-lib.tar.gz "$url"
        fi
        tar -xf ta-lib.tar.gz
        cd ta-lib
        ./configure --prefix=/usr/local
        make -j"$(nproc 2>/dev/null || echo 1)"
        sudo make install
        sudo ldconfig 2>/dev/null || true
        cd /
        rm -rf "$tmp"
    fi
}

if ta_lib_exists; then
    echo "[1/3] 已检测到系统已安装 ta-lib C 库，跳过编译。"
else
    install_ta_lib
fi

echo "[3/3] 安装 TA-Lib python 包 (公网 PyPI) ..."
"$python" -m pip install TA-Lib==0.6.4

echo
echo "=========================================================="
echo " 验证 import talib ..."
echo "=========================================================="
"$python" -c "import talib; print('talib', talib.__version__, 'OK')" || {
    echo "[失败] import talib 仍不可用，请查看上方报错"
    exit 1
}
echo
echo "TA-Lib 安装成功！现在可以正常 import vnpy 了。"
