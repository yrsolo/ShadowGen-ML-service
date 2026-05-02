@echo off
setlocal EnableExtensions

if "%~1"=="-?" goto help
if "%~1"=="/?" goto help
if /I "%~1"=="help" goto help

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if "%TRITON_IMAGE%"=="" set "TRITON_IMAGE=shadowgen-triton-segmenter:py"
if "%TRITON_CONTAINER%"=="" set "TRITON_CONTAINER=shadowgen-triton-segmenter"
if "%NO_CACHE%"=="" set "NO_CACHE=0"

set "DOCKER_EXE=docker"

echo Rebuilding ShadowGen Triton image with baked model repository.
echo Image:     %TRITON_IMAGE%
echo Container: %TRITON_CONTAINER%
echo.

"%DOCKER_EXE%" version --format "{{.Server.Version}}" >nul
if errorlevel 1 (
  echo [ERROR] Docker daemon is not reachable. Start Docker Desktop and retry.
  exit /b 1
)

for /f "delims=" %%A in ('%DOCKER_EXE% ps -a --filter "name=^/%TRITON_CONTAINER%$" --format "{{.Names}}"') do set "EXISTING_CONTAINER=%%A"
if /I "%EXISTING_CONTAINER%"=="%TRITON_CONTAINER%" (
  echo Removing old Triton container so the next service start uses the rebuilt image...
  "%DOCKER_EXE%" rm -f "%TRITON_CONTAINER%" >nul
)

set "BUILD_CACHE_ARG="
if "%NO_CACHE%"=="1" set "BUILD_CACHE_ARG=--no-cache"
if /I "%NO_CACHE%"=="true" set "BUILD_CACHE_ARG=--no-cache"
if /I "%NO_CACHE%"=="yes" set "BUILD_CACHE_ARG=--no-cache"

"%DOCKER_EXE%" build %BUILD_CACHE_ARG% -f "%ROOT_DIR%ops\triton\Dockerfile.segmenter-python" -t "%TRITON_IMAGE%" "%ROOT_DIR%ops\triton"
if errorlevel 1 (
  echo [ERROR] Docker build failed.
  exit /b 1
)

echo.
echo Triton image rebuilt successfully.
echo Next step:
echo   start-service.cmd
exit /b 0

:help
echo Usage:
echo   rebuild-triton.cmd
echo.
echo Rebuilds the Triton Docker image with the current ops\triton\model_repository code baked into /models.
echo It also removes the old Triton container so the next service start uses the new image.
echo.
echo Optional environment variables:
echo   TRITON_IMAGE=shadowgen-triton-segmenter:py
echo   TRITON_CONTAINER=shadowgen-triton-segmenter
echo   NO_CACHE=1
exit /b 0
