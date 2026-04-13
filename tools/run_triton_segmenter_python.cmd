@echo off
setlocal

if "%~1"=="-?" goto help
if "%~1"=="/?" goto help
if /I "%~1"=="help" goto help

set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_triton_segmenter_python.ps1" %*
exit /b %ERRORLEVEL%

:help
echo Usage:
echo   tools\run_triton_segmenter_python.cmd [-Gpu] [-BindModelRepository] [-Detach] [-NoBuild] [-HttpPort 8010] [-GrpcPort 8011] [-MetricsPort 8012]
echo.
echo Starts a local Triton container for shadowgen_segmenter.
echo Default host ports avoid FastAPI's local 8000:
echo   HTTP    http://127.0.0.1:8010
echo   gRPC    127.0.0.1:8011
echo   metrics http://127.0.0.1:8012/metrics
echo.
echo By default the container starts without Docker GPU flags for dev bring-up.
echo Use -Gpu when Docker Desktop NVIDIA GPU support is configured.
echo By default the model repository is baked into the image to avoid Windows bind-mount issues.
echo Use -BindModelRepository only when Docker can mount this workspace path reliably.
echo Use -Detach to run Triton in the background.
exit /b 0
