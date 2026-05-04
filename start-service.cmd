@echo off
setlocal EnableExtensions

if "%~1"=="-?" goto help
if "%~1"=="/?" goto help
if /I "%~1"=="help" goto help

if /I not "%~1"=="--foreground" (
  start "ShadowGen ML Service" cmd /k ""%~f0" --foreground %*"
  exit /b 0
)

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: "%PYTHON_EXE%"
  echo Create the virtual environment and install dependencies first.
  exit /b 1
)

if "%HOST%"=="" set "HOST=127.0.0.1"
if "%PORT%"=="" set "PORT=8000"
if "%RELOAD%"=="" set "RELOAD=0"
if "%TRITON_HTTP_PORT%"=="" set "TRITON_HTTP_PORT=8010"
if "%USE_TRITON_BACKENDS%"=="" set "USE_TRITON_BACKENDS=0"

if "%SHADOWGEN_TRITON_URL%"=="" set "SHADOWGEN_TRITON_URL=http://127.0.0.1:%TRITON_HTTP_PORT%"
if "%SHADOWGEN_TRITON_TRANSPORT%"=="" set "SHADOWGEN_TRITON_TRANSPORT=native"
if "%USE_TRITON_BACKENDS%"=="1" (
  if "%SHADOWGEN_DETECTOR_BACKEND_KIND%"=="" set "SHADOWGEN_DETECTOR_BACKEND_KIND=triton"
  if "%SHADOWGEN_SEGMENTER_BACKEND_KIND%"=="" set "SHADOWGEN_SEGMENTER_BACKEND_KIND=triton"
) else (
  if "%SHADOWGEN_DETECTOR_BACKEND_KIND%"=="" set "SHADOWGEN_DETECTOR_BACKEND_KIND=local"
  if "%SHADOWGEN_SEGMENTER_BACKEND_KIND%"=="" set "SHADOWGEN_SEGMENTER_BACKEND_KIND=local"
)

set "RELOAD_ARG="
if "%RELOAD%"=="1" set "RELOAD_ARG=--reload"
if /I "%RELOAD%"=="true" set "RELOAD_ARG=--reload"
if /I "%RELOAD%"=="yes" set "RELOAD_ARG=--reload"

echo ShadowGen ML Service launcher
echo FastAPI: http://%HOST%:%PORT%/playground
echo Triton:  %SHADOWGEN_TRITON_URL%
echo Backends: detector=%SHADOWGEN_DETECTOR_BACKEND_KIND%, segmenter=%SHADOWGEN_SEGMENTER_BACKEND_KIND%
echo.
echo This script does not start Triton. Use start-triton.cmd for the Triton container.
echo Press Ctrl+C in this window or use the Playground shutdown button to stop FastAPI.
echo.
"%PYTHON_EXE%" -m uvicorn shadowgen_ml_service.main:app --host %HOST% --port %PORT% %RELOAD_ARG%
exit /b %ERRORLEVEL%

:help
echo Usage:
echo   start-service.cmd
echo.
echo Starts only the FastAPI ML service in a visible console window.
echo Start Triton separately when needed:
echo   start-triton.cmd
echo.
echo Optional environment variables:
echo   PORT=8000
echo   HOST=127.0.0.1
echo   RELOAD=0
echo   TRITON_HTTP_PORT=8010
echo   USE_TRITON_BACKENDS=0
echo   SHADOWGEN_TRITON_TRANSPORT=native
echo   SHADOWGEN_DETECTOR_MODEL_VARIANT=grounding-dino
echo   SHADOWGEN_SEGMENTER_MODEL_VARIANT=birefnet
exit /b 0
