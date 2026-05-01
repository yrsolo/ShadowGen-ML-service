@echo off
setlocal

set "ROOT_DIR=%~dp0"
if "%HOST%"=="" set "HOST=127.0.0.1"
if "%PORT%"=="" set "PORT=8000"
if "%RELOAD%"=="" set "RELOAD=0"

echo Opening visible ShadowGen ML Service console on http://%HOST%:%PORT%
echo Close that window or use the Playground shutdown button to stop the service.
start "ShadowGen ML Service :%PORT%" cmd /k ""%ROOT_DIR%run-service.cmd""

endlocal
