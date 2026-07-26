@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

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
    -r requirements-lock.txt
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
    --hidden-import app.ai.video_test ^
    --collect-all fastapi ^
    --collect-all starlette ^
    --collect-all uvicorn ^
    --collect-all cv2 ^
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

echo Portable build completed successfully.
echo %CD%\dist\BCVision\BCVision.exe
exit /b 0

:error
echo Portable build failed.
exit /b 1
