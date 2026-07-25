@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
where winget >nul 2>&1 || (echo winget در دسترس نیست.& pause & exit /b 1)
where py >nul 2>&1 || winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
if not exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" if not exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" winget install --id JRSoftware.InnoSetup -e --silent --accept-package-agreements --accept-source-agreements
call BUILD_PORTABLE_EXE.bat
if errorlevel 1 exit /b 1
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
"%ISCC%" installer\BCVision.iss
if errorlevel 1 (echo ساخت Setup ناموفق بود.& pause & exit /b 1)
echo فایل نصبی در setup_output ساخته شد.
pause
