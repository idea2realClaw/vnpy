@ECHO OFF
SETLOCAL
SET python=%1
IF "%python%"=="" SET python=python

ECHO ==========================================================
ECHO  TA-Lib C 库安装脚本 (Windows)
ECHO  仓库: idea2realClaw/vnpy
ECHO ==========================================================
ECHO.

:: 0) 升级 pip / wheel
ECHO [0/3] 升级 pip / wheel ...
%python% -m pip install --upgrade pip wheel
IF ERRORLEVEL 1 (ECHO [错误] pip 升级失败 & EXIT /B 1)

:: 1) 优先：使用 VeighNa 私有源里的预编译 ta_lib 轮子（无需单独装 C 库）
ECHO [1/3] 尝试从 pypi.vnpy.com 安装预编译 ta_lib 轮子 ...
%python% -m pip install ta_lib==0.6.4 --extra-index-url https://pypi.vnpy.com
IF %ERRORLEVEL%==0 (
    ECHO [完成] ta_lib 已通过预编译轮子装好，import talib 应该可以直接用。
    GOTO :verify
)

:: 2) 回退：安装官方 TA-Lib C 库 (MSVC 预编译) 到 C:\ta-lib，再从公网装 python 包
ECHO [2/3] 预编译轮子不可用，改为安装官方 TA-Lib C 库 ...
SET TA_LIB_DIR=C:\ta-lib
SET TA_URL=https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-msvc.zip
SET TA_ZIP=%TEMP%\ta-lib-0.4.0-msvc.zip

IF EXIST "%TA_LIB_DIR%\c\include\ta_lib.h" (
    ECHO [跳过] 已检测到 %TA_LIB_DIR%\c\include\ta_lib.h，C 库似乎已存在。
) ELSE (
    ECHO 正在下载 TA-Lib C 库 (MSVC) ...
    powershell -Command "Invoke-WebRequest -Uri '%TA_URL%' -OutFile '%TA_ZIP%'"
    IF NOT EXIST "%TA_ZIP%" (ECHO [错误] 下载失败 & EXIT /B 1)
    ECHO 正在解压到 %TA_LIB_DIR% ...
    powershell -Command "Expand-Archive -Path '%TA_ZIP%' -DestinationPath 'C:\' -Force"
    IF NOT EXIST "%TA_LIB_DIR%\c\include\ta_lib.h" (
        ECHO [错误] 解压后未找到 %TA_LIB_DIR%\c\include\ta_lib.h & EXIT /B 1
    )
)

:: 设置环境变量，供 TA-Lib python 包编译/运行时查找
SET TA_LIBRARY_PATH=%TA_LIB_DIR%\c\lib
SET TA_INCLUDE_PATH=%TA_LIB_DIR%\c\include
ECHO 设置 TA_LIBRARY_PATH=%TA_LIBRARY_PATH%
ECHO 设置 TA_INCLUDE_PATH=%TA_INCLUDE_PATH%

ECHO 正在安装 TA-Lib python 包 (公网 PyPI) ...
%python% -m pip install TA-Lib==0.6.4
IF ERRORLEVEL 1 (ECHO [错误] TA-Lib python 包安装失败 & EXIT /B 1)
ECHO [提示] 若运行时仍报找不到模块，请确认 C:\ta-lib 存在，或将上面两个环境变量加入系统环境变量。

:verify
ECHO [3/3] 验证 import talib ...
%python% -c "import talib; print('talib', talib.__version__, 'OK')"
IF ERRORLEVEL 1 (ECHO [失败] import talib 仍不可用，请查看上方报错 & EXIT /B 1)

ECHO.
ECHO ==========================================================
ECHO  TA-Lib 安装成功！现在可以正常 import vnpy 了。
ECHO ==========================================================
ENDLOCAL
