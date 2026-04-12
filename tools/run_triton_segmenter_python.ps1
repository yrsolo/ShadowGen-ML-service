$ErrorActionPreference = "Stop"

$dockerBin = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
$imageName = "shadowgen-triton-segmenter:py"
$containerName = "shadowgen-triton-segmenter"
$modelRepository = Join-Path $PSScriptRoot "..\ops\triton\model_repository"

if (-not (Test-Path $dockerBin)) {
    throw "Docker CLI not found at $dockerBin"
}

$dockerRoot = Split-Path $dockerBin -Parent
$env:PATH = "$dockerRoot;$env:PATH"

$wslList = (& wsl -l -q) 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "WSL is not available. Triton container bring-up requires a working Linux container backend."
}

if ($wslList -notmatch "docker-desktop") {
    throw "Docker Desktop WSL distro 'docker-desktop' is missing. Repair Docker Desktop before trying to run Triton."
}

Write-Host "Building Triton segmenter image..."
& $dockerBin build -f (Join-Path $PSScriptRoot "..\ops\triton\Dockerfile.segmenter-python") -t $imageName (Join-Path $PSScriptRoot "..")

Write-Host "Stopping any existing container..."
& $dockerBin rm -f $containerName 2>$null | Out-Null

Write-Host "Starting Triton container..."
& $dockerBin run --name $containerName --rm -p 8001:8001 -v "${modelRepository}:/models" $imageName tritonserver --model-repository=/models
