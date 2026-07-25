@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo.
echo ==========================================
echo   BC Vision 2.1.0 - One Click Update
echo ==========================================
echo.
set "TARGET=%ProgramFiles%\BC Vision"
if not exist "%TARGET%" set "TARGET=%LocalAppData%\BC Vision"
if not exist "%TARGET%" (
  echo محل نصب BC Vision پیدا نشد.
  echo پوشه نصب را کنار این فایل با نام BC Vision قرار دهید یا RUN_SOURCE.bat را اجرا کنید.
  pause
  exit /b 1
)
echo Updating: %TARGET%
xcopy "%~dp0app" "%TARGET%\app\" /E /I /Y >nul
copy /Y "%~dp0launcher.py" "%TARGET%\launcher.py" >nul
copy /Y "%~dp0VERSION" "%TARGET%\VERSION" >nul
copy /Y "%~dp0requirements.txt" "%TARGET%\requirements.txt" >nul
copy /Y "%~dp0license_public_key.pem" "%TARGET%\license_public_key.pem" >nul
echo.
echo بروزرسانی با موفقیت انجام شد.
echo اطلاعات دیتابیس و تنظیمات حذف نشده‌اند.
pause
