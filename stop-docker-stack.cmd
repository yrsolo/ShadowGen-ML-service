@echo off
setlocal
cd /d "%~dp0"

if "%TRITON_CONTAINER%"=="" set "TRITON_CONTAINER=shadowgen-triton-segmenter"
if "%SERVICE_CONTAINER%"=="" set "SERVICE_CONTAINER=shadowgen-ml-service"

echo Stopping ShadowGen docker stack...
docker compose down
docker rm -f "%TRITON_CONTAINER%" >nul 2>nul
docker rm -f "%SERVICE_CONTAINER%" >nul 2>nul
