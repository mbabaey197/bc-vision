@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=
where py >nul 2>&1 && set PY=py
if not defined PY (where python >nul 2>&1 && set PY=python)
if not defined PY (
 echo Python 3.11 or 3.12 is not installed.
 pause
 exit /b 1
)
if not exist .venv (
 %PY% -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error
python launcher.py
exit /b
:error
echo.
echo Installation or execution failed.
pause
