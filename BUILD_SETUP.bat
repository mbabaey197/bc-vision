@echo off
chcp 65001 >nul
cd /d "%~dp0"
call BUILD_PORTABLE_EXE.bat
if errorlevel 1 exit /b 1
set ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe
if not exist "%ISCC%" set ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe
if not exist "%ISCC%" (
 echo Inno Setup 6 is not installed.
 echo Install it, then run this file again.
 pause
 exit /b 1
)
"%ISCC%" installer\BCVision.iss
echo Setup created in setup_output
pause
