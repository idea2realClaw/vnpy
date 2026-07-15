# 安装 TA-Lib C 库

VeighNa（vnpy）的核心指标计算依赖 [TA-Lib](https://ta-lib.org/)：

- `vnpy/trader/utility.py` 中的 `ArrayManager` 使用 `talib` 实现了约 40 个技术指标
  （SMA / EMA / KAMA / RSI / MACD / ADX / ADXR / ATR / NATR / CCI / BOLL / Donchian /
  SAR / STOCH / AROON / MFI / OBV 等，代码位于 `utility.py` 第 590–1267 行）；
- `vnpy/alpha/dataset/ta_function.py` 用 `talib` 计算 RSI、ATR 等因子。

TA-Lib 由两部分组成：

1. **C 语言底层库**（`libta_lib`，系统级，需编译 / 安装到系统路径）；
2. **Python 封装包**（`TA-Lib` / 预编译 `ta_lib` 轮子，提供 `import talib`）。

如果只装了 Python 包而系统没有 C 库，`import talib` 会直接失败，
进而导致 `import vnpy` 在 `utility.py` 顶部的 `import talib` 处崩溃。
本仓库自带的 `install.bat` / `install.sh` / `install_osx.sh` 已包含 TA-Lib 安装逻辑，
但都依赖 VeighNa 私有源 `pypi.vnpy.com` 提供的预编译轮子。
下面这套独立脚本在你**无法访问私有源**时仍可把 C 库装好。

## 快速使用

### Windows

```bat
install_talib.bat                :: 可选传入 python 解释器，默认 python
install_talib.bat python3.12
```

脚本逻辑：

1. 先尝试从 `pypi.vnpy.com` 安装预编译 `ta_lib==0.6.4` 轮子（最省事）；
2. 若私有源不可用，则下载官方 MSVC 预编译 C 库到 `C:\ta-lib`，
   设置 `TA_LIBRARY_PATH` / `TA_INCLUDE_PATH` 后从公网 PyPI 安装 `TA-Lib==0.6.4`。

### Linux

```bash
bash install_talib.sh            :: 可选传入 python，默认 python3
bash install_talib.sh python3.12
```

脚本逻辑：

1. 检测系统是否已存在 `ta-lib` C 库（存在则跳过编译）；
2. 否则下载 0.4.0 源码编译安装到 `/usr/local`；
3. 安装 `TA-Lib==0.6.4` Python 包并验证 `import talib`。

### macOS

```bash
bash install_talib.sh
```

脚本会通过 `brew install ta-lib` 安装 C 库，再装 Python 包。

## 验证

```bash
python -c "import talib; print(talib.__version__)"
# 0.6.4 OK
```

## 排错

- **`No module named 'talib'`**：Python 包没装好，重新运行上面的脚本。
- **编译时找不到 `ta_lib.h` / `libta_lib`**：确认 C 库已装到系统路径；
  手动设置环境变量（Windows 解压到 `C:\ta-lib` 后一般无需设置）：
  ```bash
  export TA_LIBRARY_PATH=/usr/local/lib
  export TA_INCLUDE_PATH=/usr/local/include
  ```
- **Windows 下载慢 / 失败**：可手动从
  https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-msvc.zip
  下载并解压到 `C:\ta-lib`，再执行 `pip install TA-Lib==0.6.4`。
