@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"

set "PY="
py -3.13 --version >nul 2>&1 && set "PY=py -3.13"

if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)

if not defined PY (
    echo Python 3.13 is required.
    exit /b 1
)

if not exist "requirements-lock.txt" (
    echo requirements-lock.txt was not found.
    exit /b 1
)

if not exist "requirements-ai-lock.txt" (
    echo requirements-ai-lock.txt was not found.
    exit /b 1
)

if not exist "license_public_key.pem" (
    echo license_public_key.pem was not found.
    exit /b 1
)

if not exist ".buildvenv\Scripts\python.exe" (
    %PY% -m venv ".buildvenv"
    if errorlevel 1 goto :error
)

set "BUILD_PY=%CD%\.buildvenv\Scripts\python.exe"

"%BUILD_PY%" -m pip install ^
    --disable-pip-version-check ^
    -r requirements-lock.txt ^
    -r requirements-ai-lock.txt
if errorlevel 1 goto :error

"%BUILD_PY%" -m pip check
if errorlevel 1 goto :error

rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

"%BUILD_PY%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --windowed ^
    --onedir ^
    --name BCVision ^
    --hidden-import app.main ^
    --hidden-import app.database ^
    --hidden-import app.security ^
    --hidden-import app.config ^
    --hidden-import app.streams ^
    --hidden-import app.license ^
    --hidden-import app.ai.detector ^
    --hidden-import app.ai.ocr ^
    --hidden-import app.ai.pipeline ^
    --hidden-import app.ai.plate_rules ^
    --hidden-import app.ai.vehicle_intelligence ^
    --hidden-import app.ai.video_test ^
    --hidden-import app.ai.live_worker ^
    --hidden-import app.ai.model_manager ^
    --collect-all fastapi ^
    --collect-all starlette ^
    --collect-all uvicorn ^
    --collect-all cv2 ^
    --collect-all easyocr ^
    --collect-all ultralytics ^
    --collect-all torch ^
    --collect-all torchvision ^
    --collect-all skimage ^
    --collect-all scipy ^
    launcher.py
if errorlevel 1 goto :error

if not exist "dist\BCVision\BCVision.exe" (
    echo BCVision.exe was not created.
    goto :error
)

copy /Y "license_public_key.pem" "dist\BCVision\license_public_key.pem" >nul
if errorlevel 1 goto :error

copy /Y "VERSION" "dist\BCVision\VERSION" >nul
if errorlevel 1 goto :error

copy /Y "README_FA.txt" "dist\BCVision\README_FA.txt" >nul
if errorlevel 1 goto :error

copy /Y "requirements-ai-lock.txt" "dist\BCVision\requirements-ai-lock.txt" >nul
if errorlevel 1 goto :error

"dist\BCVision\BCVision.exe" --help >nul 2>&1
rem A windowed launcher may ignore --help; existence is the required build gate.

echo Portable AI build completed successfully.
echo %CD%\dist\BCVision\BCVision.exe
exit /b 0

:error
echo Portable AI build failed.
exit /b 1
