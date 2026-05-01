@echo off
setlocal

set "ROOT_DIR=%~dp0"
if "%HOST%"=="" set "HOST=127.0.0.1"
if "%PORT%"=="" set "PORT=8003"
if "%RELOAD%"=="" set "RELOAD=0"
if "%SHADOWGEN_TRITON_URL%"=="" set "SHADOWGEN_TRITON_URL=http://127.0.0.1:8010"
if "%SHADOWGEN_SEGMENTER_BACKEND_KIND%"=="" set "SHADOWGEN_SEGMENTER_BACKEND_KIND=triton"

echo Opening visible ShadowGen ML Service console with Triton segmenter defaults.
echo Service: http://%HOST%:%PORT%
echo Triton:  %SHADOWGEN_TRITON_URL%
echo Close that window or use the Playground shutdown button to stop the service.
start "ShadowGen ML Service Triton :%PORT%" cmd /k ""%ROOT_DIR%run-service-triton-segmenter.cmd""

endlocal
