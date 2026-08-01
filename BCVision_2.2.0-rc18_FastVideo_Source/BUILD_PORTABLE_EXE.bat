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

rmdir /s /q .model-seed 2>nul
"%BUILD_PY%" -m app.ai.model_manager --seed-dir ".model-seed"
if errorlevel 1 goto :error

rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

rem RC15 operator-assisted ANPR modules are packaged explicitly.
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
    --hidden-import app.ai.onnx_crnn ^
    --hidden-import app.ai.onnx_cnn ^
    --hidden-import app.ai.onnx_detector ^
    --hidden-import app.ai.pipeline ^
    --hidden-import app.ai.plate_recovery ^
    --hidden-import app.ai.plate_rules ^
    --hidden-import app.ai.vehicle_intelligence ^
    --hidden-import app.ai.video_test ^
    --hidden-import app.ai.live_worker ^
    --hidden-import app.ai.evaluation ^
    --hidden-import app.ai.activity ^
    --hidden-import app.ai.golden ^
    --hidden-import app.ai.next_engine ^
    --hidden-import app.ai.next_models ^
    --hidden-import app.ai.onnx_cct ^
    --hidden-import app.ai.onnx_obb ^
    --hidden-import app.ai.review_policy ^
    --hidden-import app.ai.model_manager ^
    --hidden-import app.ai.training ^
    --hidden-import app.ai.training_worker ^
    --add-data ".model-seed;model-seed" ^
    --collect-all av ^
    --collect-all fastapi ^
    --collect-all starlette ^
    --collect-all uvicorn ^
    --collect-all cv2 ^
    --collect-all onnx ^
    --collect-all onnxruntime ^
    --collect-all torch ^
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

copy /Y "THIRD_PARTY_NOTICES.md" "dist\BCVision\THIRD_PARTY_NOTICES.md" >nul
if errorlevel 1 goto :error

echo Portable AI build completed successfully.
echo %CD%\dist\BCVision\BCVision.exe
exit /b 0

:error
echo Portable AI build failed.
exit /b 1
