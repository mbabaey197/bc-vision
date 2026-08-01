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
if not exist ".venv\Scripts\python.exe" (
 %PY% -m venv ".venv" || goto :error
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
if not exist ".venv\Scripts\pythonw.exe" goto :error
start "" /B "%CD%\.venv\Scripts\pythonw.exe" "%CD%\launcher.py"
exit /b 0
:error
echo.
echo Installation or execution failed.
pause
