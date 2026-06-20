@echo off
setlocal
cd /d "%~dp0"

if "%TRITON_HTTP_PORT%"=="" set "TRITON_HTTP_PORT=8010"
if "%SERVICE_HTTP_PORT%"=="" if exist ".env" (
  for /f "tokens=1,* delims==" %%A in ('findstr /B /I "SERVICE_HTTP_PORT=" ".env"') do set "SERVICE_HTTP_PORT=%%B"
)
if "%SERVICE_HTTP_PORT%"=="" set "SERVICE_HTTP_PORT=9001"
if "%TRITON_CONTAINER%"=="" set "TRITON_CONTAINER=shadowgen-triton-segmenter"
if "%SERVICE_CONTAINER%"=="" set "SERVICE_CONTAINER=shadowgen-ml-service"

echo Starting ShadowGen docker stack:
echo   - Triton:  http://127.0.0.1:%TRITON_HTTP_PORT%
echo   - Service: http://127.0.0.1:%SERVICE_HTTP_PORT%
echo.
echo Defaults are Triton 8010 and Service 9001 when env vars are not set.
echo.

docker rm -f "%TRITON_CONTAINER%" >nul 2>nul
docker rm -f "%SERVICE_CONTAINER%" >nul 2>nul

docker compose up -d
if errorlevel 1 exit /b %errorlevel%

echo.
echo Stack is starting. Open http://127.0.0.1:%SERVICE_HTTP_PORT%/playground
