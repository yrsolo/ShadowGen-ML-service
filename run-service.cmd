@echo off
setlocal

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
if "%RELOAD%"=="" set "RELOAD=1"

set "RELOAD_ARG=--reload"
if "%RELOAD%"=="0" set "RELOAD_ARG="
if /I "%RELOAD%"=="false" set "RELOAD_ARG="
if /I "%RELOAD%"=="no" set "RELOAD_ARG="

echo Starting ShadowGen ML Service on http://%HOST%:%PORT%
echo Playground UI: http://%HOST%:%PORT%/playground
"%PYTHON_EXE%" -m uvicorn shadowgen_ml_service.main:app --host %HOST% --port %PORT% %RELOAD_ARG%

endlocal
