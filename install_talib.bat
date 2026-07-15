@ECHO OFF
SETLOCAL
SET "PY=%~1"
IF "%PY%"=="" SET "PY=python"

ECHO ==========================================================
ECHO  TA-Lib C library installer (Windows)
ECHO  repo: idea2realClaw/vnpy
ECHO ==========================================================
ECHO.

ECHO [0/3] Upgrading pip / wheel ...
"%PY%" -m pip install --upgrade pip wheel
IF ERRORLEVEL 1 (ECHO [ERR] pip upgrade failed & PAUSE & EXIT /B 1)
PAUSE

ECHO [1/3] Trying prebuilt ta_lib wheel from pypi.vnpy.com ...
"%PY%" -m pip install ta_lib==0.6.4 --extra-index-url https://pypi.vnpy.com
PAUSE
IF NOT ERRORLEVEL 1 GOTO :verify

ECHO [2/3] Prebuilt wheel unavailable, installing official TA-Lib C lib (MSVC) ...
SET "TA_LIB_DIR=C:\ta-lib"
SET "TA_URL=https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-msvc.zip"
SET "TA_ZIP=%TEMP%\ta-lib-0.4.0-msvc.zip"

IF EXIST "%TA_LIB_DIR%\c\include\ta_lib.h" (
    ECHO [skip] %TA_LIB_DIR%\c\include\ta_lib.h already exists.
) ELSE (
    ECHO Downloading TA-Lib C lib (MSVC) ...
    IF EXIST "%TA_ZIP%" DEL /F /Q "%TA_ZIP%"
    curl.exe -L -o "%TA_ZIP%" "%TA_URL%"
    IF ERRORLEVEL 1 (ECHO [ERR] download failed & PAUSE & EXIT /B 1)
    IF NOT EXIST "%TA_ZIP%" (ECHO [ERR] zip file not found after download & PAUSE & EXIT /B 1)
    PAUSE
    ECHO Extracting to %TA_LIB_DIR% ...
    IF NOT EXIST "%TA_LIB_DIR%" MKDIR "%TA_LIB_DIR%"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force -Path '%TA_ZIP%' -DestinationPath '%TA_LIB_DIR%'"
    IF ERRORLEVEL 1 (ECHO [ERR] extract failed & PAUSE & EXIT /B 1)
    PAUSE
    IF NOT EXIST "%TA_LIB_DIR%\c\include\ta_lib.h" (
        FOR /D %%D IN ("%TA_LIB_DIR%\*") DO (
            IF EXIST "%%D\c\include\ta_lib.h" (
                ECHO Moving contents from %%D ...
                MOVE /Y "%%D\*" "%TA_LIB_DIR%\" >NUL
            )
        )
    )
    IF NOT EXIST "%TA_LIB_DIR%\c\include\ta_lib.h" (
        ECHO [ERR] ta_lib.h not found after extraction & PAUSE & EXIT /B 1
    )
)

SET "TA_LIBRARY_PATH=%TA_LIB_DIR%\c\lib"
SET "TA_INCLUDE_PATH=%TA_LIB_DIR%\c\include"
ECHO TA_LIBRARY_PATH=%TA_LIBRARY_PATH%
ECHO TA_INCLUDE_PATH=%TA_INCLUDE_PATH%

ECHO Installing TA-Lib python package (public PyPI) ...
"%PY%" -m pip install TA-Lib==0.6.4
IF ERRORLEVEL 1 (ECHO [ERR] TA-Lib python install failed & PAUSE & EXIT /B 1)
PAUSE
ECHO [tip] if import still fails, ensure C:\ta-lib exists, or add the two env vars above to system PATH.

:verify
ECHO [3/3] Verifying import talib ...
"%PY%" -c "import talib; print('talib', talib.__version__, 'OK')"
IF ERRORLEVEL 1 (ECHO [FAIL] import talib still unavailable, see errors above & PAUSE & EXIT /B 1)
PAUSE

ECHO.
ECHO ==========================================================
ECHO  TA-Lib installed! Now you can import vnpy normally.
ECHO ==========================================================
PAUSE
ENDLOCAL
