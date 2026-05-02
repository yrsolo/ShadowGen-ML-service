@echo off
setlocal EnableExtensions EnableDelayedExpansion

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

if "%TRITON_IMAGE%"=="" set "TRITON_IMAGE=shadowgen-triton-segmenter:py"
if "%TRITON_CONTAINER%"=="" set "TRITON_CONTAINER=shadowgen-triton-segmenter"
if "%TRITON_HTTP_PORT%"=="" set "TRITON_HTTP_PORT=8010"
if "%TRITON_GRPC_PORT%"=="" set "TRITON_GRPC_PORT=8011"
if "%TRITON_METRICS_PORT%"=="" set "TRITON_METRICS_PORT=8012"
if "%TRITON_GPU%"=="" set "TRITON_GPU=1"
if "%TRITON_MODEL_ID%"=="" set "TRITON_MODEL_ID=ZhengPeng7/BiRefNet-matting"
if "%TRITON_DETECTOR_MODEL_ID%"=="" set "TRITON_DETECTOR_MODEL_ID=IDEA-Research/grounding-dino-base"
if "%TRITON_DETECTOR_PROMPT%"=="" set "TRITON_DETECTOR_PROMPT=object."
if "%TRITON_DETECTOR_BOX_THRESHOLD%"=="" set "TRITON_DETECTOR_BOX_THRESHOLD=0.25"
if "%TRITON_DETECTOR_TEXT_THRESHOLD%"=="" set "TRITON_DETECTOR_TEXT_THRESHOLD=0.25"
if "%TRITON_RESOLUTION%"=="" set "TRITON_RESOLUTION=512"
if "%TRITON_DEVICE%"=="" (
  if "%TRITON_GPU%"=="1" (
    set "TRITON_DEVICE=cuda:0"
  ) else (
    set "TRITON_DEVICE=cpu"
  )
)
if "%TRITON_DETECTOR_DEVICE%"=="" set "TRITON_DETECTOR_DEVICE=%TRITON_DEVICE%"
if "%TRITON_COMPILE_ENABLED%"=="" set "TRITON_COMPILE_ENABLED=false"
if "%HF_CACHE_DIR%"=="" set "HF_CACHE_DIR=%LOCALAPPDATA%\ShadowGen\triton-hf-cache-gpu"
if "%STOP_TRITON_ON_EXIT%"=="" set "STOP_TRITON_ON_EXIT=0"

set "SHADOWGEN_TRITON_URL=http://127.0.0.1:%TRITON_HTTP_PORT%"
if "%SHADOWGEN_DETECTOR_BACKEND_KIND%"=="" set "SHADOWGEN_DETECTOR_BACKEND_KIND=triton"
if "%SHADOWGEN_SEGMENTER_BACKEND_KIND%"=="" set "SHADOWGEN_SEGMENTER_BACKEND_KIND=triton"

set "DOCKER_EXE=docker"

echo ShadowGen ML Service launcher
echo FastAPI: http://%HOST%:%PORT%/playground
echo Triton:  %SHADOWGEN_TRITON_URL%
echo.

"%DOCKER_EXE%" version --format "{{.Server.Version}}" >nul
if errorlevel 1 (
  echo [ERROR] Docker daemon is not reachable. Start Docker Desktop and retry.
  exit /b 1
)

"%DOCKER_EXE%" image inspect "%TRITON_IMAGE%" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Triton image not found: %TRITON_IMAGE%
  echo Run this first:
  echo   rebuild-triton.cmd
  exit /b 1
)

if not exist "%HF_CACHE_DIR%" mkdir "%HF_CACHE_DIR%"

set "RUNNING_CONTAINER="
for /f "delims=" %%A in ('%DOCKER_EXE% ps --filter "name=^/%TRITON_CONTAINER%$" --format "{{.Names}}"') do set "RUNNING_CONTAINER=%%A"
if /I "%RUNNING_CONTAINER%"=="%TRITON_CONTAINER%" (
  echo Triton container is already running: %TRITON_CONTAINER%
) else (
  set "EXISTING_CONTAINER="
  for /f "delims=" %%A in ('%DOCKER_EXE% ps -a --filter "name=^/%TRITON_CONTAINER%$" --format "{{.Names}}"') do set "EXISTING_CONTAINER=%%A"
  if /I "!EXISTING_CONTAINER!"=="%TRITON_CONTAINER%" (
    echo Removing stopped Triton container: %TRITON_CONTAINER%
    "%DOCKER_EXE%" rm -f "%TRITON_CONTAINER%" >nul
  )

  set "GPU_ARGS="
  if "%TRITON_GPU%"=="1" set "GPU_ARGS=--gpus all"

  echo Starting Triton container: %TRITON_CONTAINER%
  echo   image:      %TRITON_IMAGE%
  echo   gpu:        %TRITON_GPU%
  echo   segmenter:  %TRITON_DEVICE%, %TRITON_RESOLUTION%px, compile=%TRITON_COMPILE_ENABLED%
  echo   detector:   %TRITON_DETECTOR_DEVICE%, %TRITON_DETECTOR_MODEL_ID%
  echo   resolution: %TRITON_RESOLUTION%
  echo   compile:    %TRITON_COMPILE_ENABLED%
  echo   HTTP:       http://127.0.0.1:%TRITON_HTTP_PORT%
  "%DOCKER_EXE%" run -d --name "%TRITON_CONTAINER%" !GPU_ARGS! --shm-size 2g -p "%TRITON_HTTP_PORT%:8000" -p "%TRITON_GRPC_PORT%:8001" -p "%TRITON_METRICS_PORT%:8002" -e "HF_HOME=/root/.cache/huggingface" -e "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub" -e "SHADOWGEN_TRITON_SEGMENTER_MODEL_ID=%TRITON_MODEL_ID%" -e "SHADOWGEN_TRITON_SEGMENTER_RESOLUTION=%TRITON_RESOLUTION%" -e "SHADOWGEN_TRITON_SEGMENTER_DEVICE=%TRITON_DEVICE%" -e "SHADOWGEN_TRITON_SEGMENTER_COMPILE_ENABLED=%TRITON_COMPILE_ENABLED%" -e "SHADOWGEN_TRITON_DETECTOR_MODEL_ID=%TRITON_DETECTOR_MODEL_ID%" -e "SHADOWGEN_TRITON_DETECTOR_PROMPT=%TRITON_DETECTOR_PROMPT%" -e "SHADOWGEN_TRITON_DETECTOR_BOX_THRESHOLD=%TRITON_DETECTOR_BOX_THRESHOLD%" -e "SHADOWGEN_TRITON_DETECTOR_TEXT_THRESHOLD=%TRITON_DETECTOR_TEXT_THRESHOLD%" -e "SHADOWGEN_TRITON_DETECTOR_DEVICE=%TRITON_DETECTOR_DEVICE%" -v "%HF_CACHE_DIR%:/root/.cache/huggingface" "%TRITON_IMAGE%" tritonserver --model-repository=/models --log-verbose=1
  if errorlevel 1 (
    echo [ERROR] Failed to start Triton container.
    exit /b 1
  )
)

echo Waiting for Triton model readiness...
"%PYTHON_EXE%" "%ROOT_DIR%tools\check_triton_segmenter_ready.py" "%SHADOWGEN_TRITON_URL%" --wait-seconds 240
if errorlevel 1 (
  echo [ERROR] Triton readiness check failed.
  echo Logs:
  echo   docker logs %TRITON_CONTAINER%
  exit /b 1
)
"%PYTHON_EXE%" "%ROOT_DIR%tools\check_triton_segmenter_ready.py" "%SHADOWGEN_TRITON_URL%" shadowgen_detector --wait-seconds 240
if errorlevel 1 (
  echo [ERROR] Triton detector readiness check failed.
  echo Logs:
  echo   docker logs %TRITON_CONTAINER%
  exit /b 1
)

set "RELOAD_ARG="
if "%RELOAD%"=="1" set "RELOAD_ARG=--reload"
if /I "%RELOAD%"=="true" set "RELOAD_ARG=--reload"
if /I "%RELOAD%"=="yes" set "RELOAD_ARG=--reload"

echo.
echo Starting FastAPI ML-core.
echo Press Ctrl+C in this window or use the Playground shutdown button to stop FastAPI.
echo Triton is a detached Docker container. Stop it with:
echo   docker rm -f %TRITON_CONTAINER%
echo.
"%PYTHON_EXE%" -m uvicorn shadowgen_ml_service.main:app --host %HOST% --port %PORT% %RELOAD_ARG%
set "SERVICE_EXIT_CODE=%ERRORLEVEL%"

if "%STOP_TRITON_ON_EXIT%"=="1" (
  echo Stopping Triton container because STOP_TRITON_ON_EXIT=1...
  "%DOCKER_EXE%" rm -f "%TRITON_CONTAINER%" >nul 2>nul
)

exit /b %SERVICE_EXIT_CODE%

:help
echo Usage:
echo   start-service.cmd
echo.
echo Starts the local ShadowGen ML service in a visible console window:
echo   1. starts the prebuilt Triton Docker container if it is not running
echo   2. waits until shadowgen_detector and shadowgen_segmenter are ready
echo   3. starts FastAPI on http://127.0.0.1:8000/playground
echo.
echo Rebuild the Triton image after model-code changes:
echo   rebuild-triton.cmd
echo.
echo Optional environment variables:
echo   PORT=8000
echo   HOST=127.0.0.1
echo   RELOAD=0
echo   TRITON_IMAGE=shadowgen-triton-segmenter:py
echo   TRITON_CONTAINER=shadowgen-triton-segmenter
echo   TRITON_HTTP_PORT=8010
echo   TRITON_GPU=1
echo   TRITON_RESOLUTION=512
echo   TRITON_DEVICE=cuda:0
echo   TRITON_DETECTOR_DEVICE=cuda:0
echo   TRITON_COMPILE_ENABLED=false
echo   STOP_TRITON_ON_EXIT=0
exit /b 0
