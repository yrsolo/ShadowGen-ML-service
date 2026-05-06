@echo off
setlocal EnableExtensions EnableDelayedExpansion

if "%~1"=="-?" goto help
if "%~1"=="/?" goto help
if /I "%~1"=="help" goto help

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: "%PYTHON_EXE%"
  echo Create the virtual environment and install dependencies first.
  exit /b 1
)

if "%TRITON_IMAGE%"=="" set "TRITON_IMAGE=shadowgen-triton-segmenter:py"
if "%TRITON_CONTAINER%"=="" set "TRITON_CONTAINER=shadowgen-triton-segmenter"
if "%TRITON_HTTP_PORT%"=="" set "TRITON_HTTP_PORT=8010"
if "%TRITON_GRPC_PORT%"=="" set "TRITON_GRPC_PORT=8011"
if "%TRITON_METRICS_PORT%"=="" set "TRITON_METRICS_PORT=8012"
if "%TRITON_GPU%"=="" set "TRITON_GPU=1"
if "%TRITON_GPU_DEVICE%"=="" set "TRITON_GPU_DEVICE=1"
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
if "%TRITON_BIND_MODEL_REPOSITORY%"=="" set "TRITON_BIND_MODEL_REPOSITORY=1"
if "%TRITON_FORWARD_HF_TOKEN%"=="" set "TRITON_FORWARD_HF_TOKEN=0"
if "%TRITON_FORWARD_HF_TOKEN%"=="1" if "%HF_TOKEN%"=="" if exist "%ROOT_DIR%.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%ROOT_DIR%.env") do (
    if /I "%%A"=="HF_TOKEN" set "HF_TOKEN=%%B"
  )
)
set "HF_TOKEN_ARGS="
if "%TRITON_FORWARD_HF_TOKEN%"=="1" if not "%HF_TOKEN%"=="" set "HF_TOKEN_ARGS=-e HF_TOKEN=%HF_TOKEN% -e HUGGING_FACE_HUB_TOKEN=%HF_TOKEN%"

set "SHADOWGEN_TRITON_URL=http://127.0.0.1:%TRITON_HTTP_PORT%"
set "DOCKER_EXE=docker"

echo ShadowGen Triton launcher
echo Triton: %SHADOWGEN_TRITON_URL%
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
  if "%TRITON_GPU%"=="1" (
    if /I "%TRITON_GPU_DEVICE%"=="all" (
      set "GPU_ARGS=--gpus all"
    ) else (
      set "GPU_ARGS=--gpus ""device=%TRITON_GPU_DEVICE%"""
    )
  )
  set "GPU_VISIBILITY_ARGS="
  if "%TRITON_GPU%"=="1" if /I not "%TRITON_GPU_DEVICE%"=="all" set "GPU_VISIBILITY_ARGS=-e NVIDIA_VISIBLE_DEVICES=%TRITON_GPU_DEVICE%"
  set "MODEL_REPOSITORY_ARGS="
  if "%TRITON_BIND_MODEL_REPOSITORY%"=="1" set "MODEL_REPOSITORY_ARGS=-v %ROOT_DIR%ops\triton\model_repository:/models"

  echo Starting Triton container: %TRITON_CONTAINER%
  echo   image:      %TRITON_IMAGE%
  echo   gpu:        %TRITON_GPU%
  echo   gpu device: %TRITON_GPU_DEVICE%
  echo   models:     %ROOT_DIR%ops\triton\model_repository ^(bind=%TRITON_BIND_MODEL_REPOSITORY%^)
  echo   segmenter:  %TRITON_DEVICE%, %TRITON_RESOLUTION%px, compile=%TRITON_COMPILE_ENABLED%
  echo   detector:   %TRITON_DETECTOR_DEVICE%, %TRITON_DETECTOR_MODEL_ID%
  echo   HTTP:       http://127.0.0.1:%TRITON_HTTP_PORT%
  "%DOCKER_EXE%" run -d --name "%TRITON_CONTAINER%" !GPU_ARGS! !GPU_VISIBILITY_ARGS! --shm-size 2g -p "%TRITON_HTTP_PORT%:8000" -p "%TRITON_GRPC_PORT%:8001" -p "%TRITON_METRICS_PORT%:8002" -e "HF_HOME=/root/.cache/huggingface" -e "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub" !HF_TOKEN_ARGS! -e "SHADOWGEN_TRITON_SEGMENTER_MODEL_ID=%TRITON_MODEL_ID%" -e "SHADOWGEN_TRITON_SEGMENTER_RESOLUTION=%TRITON_RESOLUTION%" -e "SHADOWGEN_TRITON_SEGMENTER_DEVICE=%TRITON_DEVICE%" -e "SHADOWGEN_TRITON_SEGMENTER_COMPILE_ENABLED=%TRITON_COMPILE_ENABLED%" -e "SHADOWGEN_TRITON_DETECTOR_MODEL_ID=%TRITON_DETECTOR_MODEL_ID%" -e "SHADOWGEN_TRITON_DETECTOR_PROMPT=%TRITON_DETECTOR_PROMPT%" -e "SHADOWGEN_TRITON_DETECTOR_BOX_THRESHOLD=%TRITON_DETECTOR_BOX_THRESHOLD%" -e "SHADOWGEN_TRITON_DETECTOR_TEXT_THRESHOLD=%TRITON_DETECTOR_TEXT_THRESHOLD%" -e "SHADOWGEN_TRITON_DETECTOR_DEVICE=%TRITON_DETECTOR_DEVICE%" -v "%HF_CACHE_DIR%:/root/.cache/huggingface" !MODEL_REPOSITORY_ARGS! "%TRITON_IMAGE%" tritonserver --model-repository=/models --log-verbose=1
  if errorlevel 1 (
    echo [ERROR] Failed to start Triton container.
    exit /b 1
  )
)

echo Waiting for required Triton model readiness...
"%PYTHON_EXE%" "%ROOT_DIR%tools\check_triton_segmenter_ready.py" "%SHADOWGEN_TRITON_URL%" shadowgen_detector --wait-seconds 240
if errorlevel 1 goto readiness_failed
"%PYTHON_EXE%" "%ROOT_DIR%tools\check_triton_segmenter_ready.py" "%SHADOWGEN_TRITON_URL%" shadowgen_segmenter --wait-seconds 240
if errorlevel 1 goto readiness_failed

if exist "%ROOT_DIR%ops\triton\model_repository\shadowgen_detector_onnx\1\model.onnx" (
  "%PYTHON_EXE%" "%ROOT_DIR%tools\check_triton_segmenter_ready.py" "%SHADOWGEN_TRITON_URL%" shadowgen_detector_onnx --wait-seconds 240
  if errorlevel 1 goto readiness_failed
)

if exist "%ROOT_DIR%ops\triton\model_repository\shadowgen_segmenter_rmbg2\1\model.onnx" (
  "%PYTHON_EXE%" "%ROOT_DIR%tools\check_triton_segmenter_ready.py" "%SHADOWGEN_TRITON_URL%" shadowgen_segmenter_rmbg2 --wait-seconds 240
  if errorlevel 1 goto readiness_failed
)

echo.
echo Triton is ready.
echo Start FastAPI separately with:
echo   start-service.cmd
echo Stop Triton with:
echo   docker rm -f %TRITON_CONTAINER%
exit /b 0

:readiness_failed
echo [ERROR] Triton readiness check failed.
echo Logs:
echo   docker logs %TRITON_CONTAINER%
exit /b 1

:help
echo Usage:
echo   start-triton.cmd
echo.
echo Starts only the Triton Docker container and waits for model readiness.
echo Rebuild the image after model-code changes:
echo   rebuild-triton.cmd
echo.
echo Optional environment variables:
echo   TRITON_IMAGE=shadowgen-triton-segmenter:py
echo   TRITON_CONTAINER=shadowgen-triton-segmenter
echo   TRITON_HTTP_PORT=8010
echo   TRITON_GPU=1
echo   TRITON_GPU_DEVICE=1
echo   TRITON_DEVICE=cuda:0
echo   TRITON_RESOLUTION=512
echo   TRITON_BIND_MODEL_REPOSITORY=1
echo   TRITON_FORWARD_HF_TOKEN=0
exit /b 0
