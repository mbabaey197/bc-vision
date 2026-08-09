@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
echo.
echo ==========================================
echo        BC Vision - Safe One Click Update
echo ==========================================
echo.

if not exist "%~dp0VERSION" (
  echo خطا: فایل VERSION کنار این راه‌انداز وجود ندارد.
  echo از بسته رسمی همان نسخه استفاده کنید.
  pause
  exit /b 1
)

set /p "APP_VERSION="<"%~dp0VERSION"
set "RC_LABEL="
for /f "tokens=2 delims=-" %%V in ("%APP_VERSION%") do set "RC_LABEL=%%V"
set "RC_LABEL=%RC_LABEL:rc=RC%"

if not defined RC_LABEL (
  echo خطا: نسخه داخل VERSION معتبر نیست: %APP_VERSION%
  pause
  exit /b 1
)

set "UPDATER=%~dp0BCVision_%RC_LABEL%_Update.exe"
if not exist "%UPDATER%" (
  echo خطا: آپدیتر رسمی تراکنشی پیدا نشد:
  echo %UPDATER%
  echo فایل Update.exe رسمی همین نسخه را کنار این فایل قرار دهید.
  echo برای امنیت، این راه‌انداز هیچ فایلی را مستقیم کپی نمی‌کند.
  pause
  exit /b 1
)

echo اجرای آپدیتر رسمی نسخه %APP_VERSION% ...
start "" /wait "%UPDATER%"
set "UPDATE_RESULT=%ERRORLEVEL%"
if not "%UPDATE_RESULT%"=="0" (
  echo.
  echo آپدیت کامل نشد. کد خروج: %UPDATE_RESULT%
  echo نسخه قبلی توسط آپدیتر رسمی حفظ یا بازیابی شده است.
  pause
  exit /b %UPDATE_RESULT%
)

echo.
echo بروزرسانی با موفقیت انجام شد.
echo اعتبارسنجی، Self-test و فعال‌سازی توسط آپدیتر رسمی انجام شد.
exit /b 0
