param(
    [string]$ImageName = "shadowgen-triton-segmenter:py",
    [string]$ContainerName = "shadowgen-triton-segmenter",
    [int]$HttpPort = 8010,
    [int]$GrpcPort = 8011,
    [int]$MetricsPort = 8012,
    [switch]$NoBuild,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host "Usage:"
    Write-Host "  tools\run_triton_segmenter_python.ps1 [-NoBuild] [-HttpPort 8010] [-GrpcPort 8011] [-MetricsPort 8012]"
    Write-Host ""
    Write-Host "Starts a local Triton container for shadowgen_segmenter."
    Write-Host "Default host ports avoid FastAPI's local 8000:"
    Write-Host "  HTTP    http://127.0.0.1:8010"
    Write-Host "  gRPC    127.0.0.1:8011"
    Write-Host "  metrics http://127.0.0.1:8012/metrics"
    exit 0
}

$dockerBin = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$dockerfile = Join-Path $repoRoot "ops\triton\Dockerfile.segmenter-python"
$modelRepository = Resolve-Path (Join-Path $repoRoot "ops\triton\model_repository")

if (-not (Test-Path $dockerBin)) {
    throw "Docker CLI not found at $dockerBin. Install or start Docker Desktop first."
}

$dockerRoot = Split-Path $dockerBin -Parent
$env:PATH = "$dockerRoot;$env:PATH"

try {
    & $dockerBin version --format "{{.Server.Version}}" | Out-Null
} catch {
    throw "Docker daemon is not reachable. Start Docker Desktop and retry."
}

try {
    $wslList = (& wsl -l -q) 2>$null
    if ($LASTEXITCODE -eq 0 -and $wslList -notmatch "docker-desktop") {
        Write-Warning "Docker Desktop WSL distro 'docker-desktop' was not found. Docker may still work with another backend, but GPU Triton usually needs Docker Desktop + WSL2."
    }
} catch {
    Write-Warning "Could not inspect WSL distributions. Continuing because Docker daemon is reachable."
}

if (-not $NoBuild) {
    Write-Host "Building Triton segmenter image: $ImageName"
    & $dockerBin build -f $dockerfile -t $ImageName $repoRoot
}

Write-Host "Stopping any existing container: $ContainerName"
$existingContainer = (& $dockerBin ps -a --filter "name=^/${ContainerName}$" --format "{{.Names}}")
if ($existingContainer -contains $ContainerName) {
    & $dockerBin rm -f $ContainerName | Out-Null
} else {
    Write-Host "  no existing container found"
}

$gpuProbeOutput = (& $dockerBin run --rm --gpus all $ImageName python3 -c "print('gpu runtime ok')" 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "Docker GPU runtime probe failed. Ensure Docker Desktop has NVIDIA GPU support enabled. Output: $gpuProbeOutput"
}

Write-Host "Starting Triton container:"
Write-Host "  image:      $ImageName"
Write-Host "  models:     $modelRepository -> /models"
Write-Host "  HTTP:       http://127.0.0.1:$HttpPort"
Write-Host "  gRPC:       127.0.0.1:$GrpcPort"
Write-Host "  metrics:    http://127.0.0.1:$MetricsPort/metrics"
Write-Host ""
Write-Host "After readiness, point ML-core to:"
Write-Host "  `$env:SHADOWGEN_TRITON_URL='http://127.0.0.1:$HttpPort'"
Write-Host "  `$env:SHADOWGEN_SEGMENTER_BACKEND_KIND='triton'"
Write-Host ""

& $dockerBin run `
    --name $ContainerName `
    --rm `
    --gpus all `
    -p "${HttpPort}:8000" `
    -p "${GrpcPort}:8001" `
    -p "${MetricsPort}:8002" `
    -v "${modelRepository}:/models" `
    $ImageName `
    tritonserver --model-repository=/models --log-verbose=1
