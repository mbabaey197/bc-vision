@echo off
setlocal
chcp 65001 >nul
net session >nul 2>&1
if not "%errorlevel%"=="0" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
set "APPDIR=%ProgramFiles%\BC Vision"
reg delete "HKLM\Software\Microsoft\Windows\CurrentVersion\Run" /v "BCVision" /f >nul 2>&1
taskkill /F /IM pythonw.exe >nul 2>&1
del /Q "%Public%\Desktop\BC Vision.lnk" >nul 2>&1
del /Q "%ProgramData%\Microsoft\Windows\Start Menu\Programs\BC Vision.lnk" >nul 2>&1
rmdir /S /Q "%APPDIR%" >nul 2>&1
echo BC Vision حذف شد.
echo دیتابیس و تصاویر شما در %ProgramData%\BC Vision باقی مانده‌اند.
pause
