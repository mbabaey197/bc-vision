@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=
where py >nul 2>&1 && set PY=py
if not defined PY (where python >nul 2>&1 && set PY=python)
if not defined PY (
 echo Install Python 3.11 or 3.12 first.
 pause
 exit /b 1
)
if not exist .buildvenv %PY% -m venv .buildvenv
call .buildvenv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul
pyinstaller --noconfirm --clean --windowed --onedir ^
 --name BCVision ^
 --hidden-import app.main ^
 --hidden-import app.database ^
 --hidden-import app.security ^
 --hidden-import app.config ^
 --hidden-import app.streams ^
 --collect-all fastapi ^
 --collect-all starlette ^
 --collect-all uvicorn ^
 --collect-all cv2 ^
 launcher.py || goto :error
if not exist dist\BCVision\data mkdir dist\BCVision\data
copy README_FA.txt dist\BCVision\README_FA.txt >nul
echo.
echo Portable EXE created:
echo %cd%\dist\BCVision\BCVision.exe
pause
exit /b
:error
echo Build failed.
pause
