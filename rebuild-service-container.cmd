@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="-?" goto help
if "%~1"=="/?" goto help
if /I "%~1"=="help" goto help

set "SERVICE_IMAGE=%SERVICE_IMAGE%"
if "%SERVICE_IMAGE%"=="" set "SERVICE_IMAGE=shadowgen-ml-service:local"

docker version --format "{{.Server.Version}}" >nul
if errorlevel 1 (
  echo [ERROR] Docker daemon is not reachable. Start Docker Desktop and retry.
  exit /b 1
)

echo Building ML service image: %SERVICE_IMAGE%
docker build -f Dockerfile.service -t "%SERVICE_IMAGE%" .
if errorlevel 1 exit /b %errorlevel%

echo Service image is ready: %SERVICE_IMAGE%
exit /b 0

:help
echo Usage:
echo   rebuild-service-container.cmd
echo.
echo Rebuilds the ML service Docker image from Dockerfile.service.
echo.
echo Optional environment variables:
echo   SERVICE_IMAGE=shadowgen-ml-service:local
exit /b 0
