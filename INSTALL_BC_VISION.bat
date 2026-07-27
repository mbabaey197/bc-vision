@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

net session >nul 2>&1
if not "%errorlevel%"=="0" (
  echo درخواست دسترسی Administrator...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

set "APPDIR=%ProgramFiles%\BC Vision"
set "DATADIR=%ProgramData%\BCVision"
set "LOGFILE=%TEMP%\BCVision_Install.log"
echo BC Vision installer started > "%LOGFILE%"

echo [1/6] آماده‌سازی پوشه‌ها...
if not exist "%APPDIR%" mkdir "%APPDIR%"
if not exist "%DATADIR%\data" mkdir "%DATADIR%\data"
if not exist "%DATADIR%\logs" mkdir "%DATADIR%\logs"
if not exist "%DATADIR%\backups" mkdir "%DATADIR%\backups"

rem Keep customer data during upgrade
if exist "%APPDIR%\data\bcvision.db" if not exist "%DATADIR%\data\bcvision.db" copy /Y "%APPDIR%\data\bcvision.db" "%DATADIR%\data\bcvision.db" >> "%LOGFILE%" 2>&1

echo [2/6] کپی فایل‌های برنامه...
robocopy "%~dp0" "%APPDIR%" /E /NFL /NDL /NJH /NJS /NP /XD .buildvenv dist build setup_output data /XF INSTALL_BC_VISION.bat UNINSTALL_BC_VISION.bat BUILD_SETUP_EXE.bat >nul
if errorlevel 8 goto :error

rem Junction keeps app compatible while storing mutable data in ProgramData
if exist "%APPDIR%\data" rmdir /S /Q "%APPDIR%\data" >nul 2>&1
mklink /J "%APPDIR%\data" "%DATADIR%\data" >> "%LOGFILE%" 2>&1

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
  echo [3/6] نصب Python 3.12...
  where winget >nul 2>&1 || (
    echo Python روی سیستم نصب نیست و winget نیز در دسترس نیست.
    echo ابتدا Python 3.11 یا 3.12 نسخه 64 بیتی را نصب کنید.
    goto :error
  )
  winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements >> "%LOGFILE%" 2>&1
  set "PATH=%LocalAppData%\Programs\Python\Python312;%LocalAppData%\Programs\Python\Python312\Scripts;%PATH%"
  set "PY=python"
) else (
  echo [3/6] Python موجود است.
)

echo [4/6] نصب موتور و پیش‌نیازها...
if not exist "%APPDIR%\.venv\Scripts\python.exe" %PY% -m venv "%APPDIR%\.venv" >> "%LOGFILE%" 2>&1
if not exist "%APPDIR%\.venv\Scripts\python.exe" goto :error
"%APPDIR%\.venv\Scripts\python.exe" -m pip install --upgrade pip >> "%LOGFILE%" 2>&1
"%APPDIR%\.venv\Scripts\python.exe" -m pip install -r "%APPDIR%\requirements.txt" >> "%LOGFILE%" 2>&1
if errorlevel 1 goto :error


echo [5/6] ساخت میانبرها و اجرای خودکار...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "$ws=New-Object -ComObject WScript.Shell;" ^
 "$s=$ws.CreateShortcut([Environment]::GetFolderPath('CommonDesktopDirectory')+'\BC Vision.lnk');" ^
 "$s.TargetPath='%APPDIR%\.venv\Scripts\pythonw.exe';" ^
 "$s.Arguments='\"%APPDIR%\launcher.py\"';" ^
 "$s.WorkingDirectory='%APPDIR%';$s.Save();" ^
 "$m=$ws.CreateShortcut($env:ProgramData+'\Microsoft\Windows\Start Menu\Programs\BC Vision.lnk');" ^
 "$m.TargetPath='%APPDIR%\.venv\Scripts\pythonw.exe';$m.Arguments='\"%APPDIR%\launcher.py\"';$m.WorkingDirectory='%APPDIR%';$m.Save();" >> "%LOGFILE%" 2>&1

reg add "HKLM\Software\Microsoft\Windows\CurrentVersion\Run" /v "BCVision" /t REG_SZ /d "\"%APPDIR%\.venv\Scripts\pythonw.exe\" \"%APPDIR%\launcher.py\"" /f >> "%LOGFILE%" 2>&1

copy /Y "%~dp0UNINSTALL_BC_VISION.bat" "%APPDIR%\UNINSTALL_BC_VISION.bat" >nul

echo [6/6] تست اجرای اولیه...
start "BC Vision" "%APPDIR%\.venv\Scripts\pythonw.exe" "%APPDIR%\launcher.py"

echo.
echo ==============================================
echo BC Vision v2.2.0-rc5 با موفقیت نصب شد.
echo نام کاربری: admin
echo رمز اولیه: 123456
echo اطلاعات برنامه در این مسیر حفظ می‌شود:
echo %DATADIR%
echo ==============================================
pause
exit /b 0

:error
echo.
echo نصب کامل نشد. گزارش خطا:
echo %LOGFILE%
pause
exit /b 1
