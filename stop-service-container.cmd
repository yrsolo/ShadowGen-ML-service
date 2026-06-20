@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%SERVICE_CONTAINER%"=="" set "SERVICE_CONTAINER=shadowgen-ml-service"

echo Stopping ShadowGen ML service container...
docker compose -f docker-compose.service.yml down
docker rm -f "%SERVICE_CONTAINER%" >nul 2>nul
