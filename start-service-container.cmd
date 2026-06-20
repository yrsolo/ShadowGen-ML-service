@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="-?" goto help
if "%~1"=="/?" goto help
if /I "%~1"=="help" goto help

if "%SERVICE_IMAGE%"=="" set "SERVICE_IMAGE=shadowgen-ml-service:local"
if "%SERVICE_CONTAINER%"=="" set "SERVICE_CONTAINER=shadowgen-ml-service"
if "%SERVICE_HTTP_PORT%"=="" if exist ".env" (
  for /f "tokens=1,* delims==" %%A in ('findstr /B /I "SERVICE_HTTP_PORT=" ".env"') do set "SERVICE_HTTP_PORT=%%B"
)
if "%SERVICE_HTTP_PORT%"=="" set "SERVICE_HTTP_PORT=9001"
if "%SERVICE_GPU_DEVICE%"=="" if exist ".env" (
  for /f "tokens=1,* delims==" %%A in ('findstr /B /I "SERVICE_GPU_DEVICE=" ".env"') do set "SERVICE_GPU_DEVICE=%%B"
)
if "%SERVICE_GPU_DEVICE%"=="" set "SERVICE_GPU_DEVICE=1"
if "%SHADOWGEN_TARGET_DEVICE%"=="" set "SHADOWGEN_TARGET_DEVICE=cuda:0"

echo ShadowGen ML service container launcher
echo   image:      %SERVICE_IMAGE%
echo   container:  %SERVICE_CONTAINER%
echo   port:       %SERVICE_HTTP_PORT%
echo   host GPU:   %SERVICE_GPU_DEVICE%
echo   app device: %SHADOWGEN_TARGET_DEVICE%
echo.
echo Inside the container the selected host GPU is addressed as cuda:0.
echo.

docker version --format "{{.Server.Version}}" >nul
if errorlevel 1 (
  echo [ERROR] Docker daemon is not reachable. Start Docker Desktop and retry.
  exit /b 1
)

docker image inspect "%SERVICE_IMAGE%" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Service image not found: %SERVICE_IMAGE%
  echo Build it first:
  echo   rebuild-service-container.cmd
  exit /b 1
)

docker rm -f "%SERVICE_CONTAINER%" >nul 2>nul

docker compose -f docker-compose.service.yml up -d
if errorlevel 1 exit /b %errorlevel%

echo.
echo Waiting for service health...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddMinutes(5); do { try { $r=Invoke-RestMethod -Uri 'http://127.0.0.1:%SERVICE_HTTP_PORT%/health' -TimeoutSec 5; if ($r.status) { Write-Host ('Service is healthy: ' + ($r | ConvertTo-Json -Compress)); exit 0 } } catch { Start-Sleep -Seconds 5 } } while ((Get-Date) -lt $deadline); Write-Error 'Service health check timed out'; exit 1"
if errorlevel 1 (
  echo [ERROR] Service did not become healthy.
  echo Logs:
  echo   docker logs %SERVICE_CONTAINER%
  exit /b 1
)

echo.
echo Service is ready:
echo   http://127.0.0.1:%SERVICE_HTTP_PORT%/health
echo   http://127.0.0.1:%SERVICE_HTTP_PORT%/v1/capabilities
exit /b 0

:help
echo Usage:
echo   start-service-container.cmd
echo.
echo Starts only the ML service container, without Triton.
echo Build the image first:
echo   rebuild-service-container.cmd
echo.
echo Useful environment variables:
echo   SERVICE_GPU_DEVICE=1  ^(normally configured in .env^)
echo   SERVICE_HTTP_PORT=9001  ^(normally configured in .env^)
echo   SERVICE_IMAGE=shadowgen-ml-service:local
echo   SERVICE_CONTAINER=shadowgen-ml-service
echo   SHADOWGEN_TARGET_DEVICE=cuda:0
echo   SHADOWGEN_SHADOW_MODEL_VARIANT=v2-diff
echo   SHADOWGEN_DEV_API_ENABLED=0
echo.
echo Configure these values in .env:
echo   SERVICE_GPU_DEVICE=1
echo   SERVICE_HTTP_PORT=9001
echo Then run:
echo   start-service-container.cmd
exit /b 0
