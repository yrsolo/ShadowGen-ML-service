param(
    [string]$ImageName = "shadowgen-triton-segmenter:py",
    [string]$ContainerName = "shadowgen-triton-segmenter",
    [int]$HttpPort = 8010,
    [int]$GrpcPort = 8011,
    [int]$MetricsPort = 8012,
    [switch]$NoBuild,
    [switch]$Gpu,
    [switch]$BindModelRepository,
    [switch]$Detach,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host "Usage:"
    Write-Host "  tools\run_triton_segmenter_python.ps1 [-Gpu] [-BindModelRepository] [-Detach] [-NoBuild] [-HttpPort 8010] [-GrpcPort 8011] [-MetricsPort 8012]"
    Write-Host ""
    Write-Host "Starts a local Triton container for shadowgen_segmenter."
    Write-Host "Default host ports avoid FastAPI's local 8000:"
    Write-Host "  HTTP    http://127.0.0.1:8010"
    Write-Host "  gRPC    127.0.0.1:8011"
    Write-Host "  metrics http://127.0.0.1:8012/metrics"
    Write-Host ""
    Write-Host "By default the container starts without Docker GPU flags for dev bring-up."
    Write-Host "Use -Gpu when Docker Desktop NVIDIA GPU support is configured."
    Write-Host "By default the model repository is baked into the image to avoid Windows bind-mount issues."
    Write-Host "Use -BindModelRepository only when Docker can mount this workspace path reliably."
    Write-Host "Use -Detach to run Triton in the background."
    exit 0
}

$dockerBin = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$tritonRoot = Resolve-Path (Join-Path $repoRoot "ops\triton")
$dockerfile = Join-Path $tritonRoot "Dockerfile.segmenter-python"
$modelRepository = Resolve-Path (Join-Path $tritonRoot "model_repository")

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

$dockerContext = (& $dockerBin context show)
Write-Host "Docker context: $dockerContext"

if (-not $NoBuild) {
    Write-Host "Building Triton segmenter image: $ImageName"
    & $dockerBin build -f $dockerfile -t $ImageName $tritonRoot
}

Write-Host "Stopping any existing container: $ContainerName"
$existingContainer = (& $dockerBin ps -a --filter "name=^/${ContainerName}$" --format "{{.Names}}")
if ($existingContainer -contains $ContainerName) {
    & $dockerBin rm -f $ContainerName | Out-Null
} else {
    Write-Host "  no existing container found"
}

$gpuArgs = @()
if ($Gpu) {
    $gpuProbeOutput = (& $dockerBin run --rm --gpus all $ImageName python3 -c "print('gpu runtime ok')" 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Docker GPU runtime probe failed. Docker Desktop is running, but NVIDIA container runtime is not available. In Docker Desktop settings, enable WSL2 backend and GPU support; also verify a CUDA-capable NVIDIA driver with 'nvidia-smi'. Output: $gpuProbeOutput"
    }
    $gpuArgs = @("--gpus", "all")
} else {
    Write-Warning "Starting Triton without Docker GPU flags. This is OK for container bring-up, but inference may run on CPU. Re-run with -Gpu after NVIDIA runtime is configured."
}

Write-Host "Starting Triton container:"
Write-Host "  image:      $ImageName"
if ($BindModelRepository) {
    Write-Host "  models:     $modelRepository -> /models (bind mount)"
} else {
Write-Host "  models:     /models (baked into image)"
}
Write-Host "  gpu:        $([bool]$Gpu)"
Write-Host "  detached:   $([bool]$Detach)"
Write-Host "  HTTP:       http://127.0.0.1:$HttpPort"
Write-Host "  gRPC:       127.0.0.1:$GrpcPort"
Write-Host "  metrics:    http://127.0.0.1:$MetricsPort/metrics"
Write-Host ""
Write-Host "After readiness, point ML-core to:"
Write-Host "  `$env:SHADOWGEN_TRITON_URL='http://127.0.0.1:$HttpPort'"
Write-Host "  `$env:SHADOWGEN_SEGMENTER_BACKEND_KIND='triton'"
Write-Host ""

$runArgs = @(
    "run",
    "--name", $ContainerName
)
if (-not $Detach) {
    $runArgs += "--rm"
}
if ($Detach) {
    $runArgs += "-d"
}
$runArgs += $gpuArgs
$runArgs += @(
    "-p", "${HttpPort}:8000",
    "-p", "${GrpcPort}:8001",
    "-p", "${MetricsPort}:8002"
)
if ($BindModelRepository) {
    $runArgs += @("-v", "${modelRepository}:/models")
}
$runArgs += @(
    $ImageName,
    "tritonserver",
    "--model-repository=/models",
    "--log-verbose=1"
)

$containerId = (& $dockerBin @runArgs)
if ($Detach) {
    Write-Host "Triton container started: $containerId"
    Write-Host "Check readiness:"
    Write-Host "  .venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:$HttpPort"
    Write-Host "Tail logs:"
    Write-Host "  docker logs -f $ContainerName"
}
