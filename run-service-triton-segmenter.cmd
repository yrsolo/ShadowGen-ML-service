@echo off
setlocal

if "%~1"=="-?" goto help
if "%~1"=="/?" goto help
if /I "%~1"=="help" goto help

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if "%SHADOWGEN_TRITON_URL%"=="" set "SHADOWGEN_TRITON_URL=http://127.0.0.1:8010"
if "%SHADOWGEN_SEGMENTER_BACKEND_KIND%"=="" set "SHADOWGEN_SEGMENTER_BACKEND_KIND=triton"
if "%RELOAD%"=="" set "RELOAD=0"

echo Starting ShadowGen ML Service with Triton segmenter
echo SHADOWGEN_TRITON_URL=%SHADOWGEN_TRITON_URL%
echo SHADOWGEN_SEGMENTER_BACKEND_KIND=%SHADOWGEN_SEGMENTER_BACKEND_KIND%
echo RELOAD=%RELOAD%
echo.

call "%ROOT_DIR%run-service.cmd"

endlocal
exit /b %ERRORLEVEL%

:help
echo Usage:
echo   run-service-triton-segmenter.cmd
echo.
echo Starts ShadowGen ML Service with:
echo   SHADOWGEN_TRITON_URL=http://127.0.0.1:8010
echo   SHADOWGEN_SEGMENTER_BACKEND_KIND=triton
echo   RELOAD=0
echo.
echo Start Triton first:
echo   tools\run_triton_segmenter_python.cmd -Detach
exit /b 0
