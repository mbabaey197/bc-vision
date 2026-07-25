@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === BCVision Diagnostic ===
echo Folder: %CD%
echo.
where py >nul 2>&1
if not errorlevel 1 (set PY=py) else (set PY=python)
%PY% --version
if errorlevel 1 goto :fail
if not exist .venv %PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail
python -c "from app.main import app; print('IMPORT OK:', app.title)"
if errorlevel 1 goto :fail
echo.
echo Starting server. Keep this window open.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
exit /b
:fail
echo.
echo Diagnostic failed. Copy or photograph the error above.
pause
