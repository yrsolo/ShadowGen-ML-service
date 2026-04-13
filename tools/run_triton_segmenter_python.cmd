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
echo   tools\run_triton_segmenter_python.cmd [-NoBuild] [-HttpPort 8010] [-GrpcPort 8011] [-MetricsPort 8012]
echo.
echo Starts a local Triton container for shadowgen_segmenter.
echo Default host ports avoid FastAPI's local 8000:
echo   HTTP    http://127.0.0.1:8010
echo   gRPC    127.0.0.1:8011
echo   metrics http://127.0.0.1:8012/metrics
exit /b 0
