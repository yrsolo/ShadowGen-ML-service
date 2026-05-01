param(
    [string]$ImageName = "shadowgen-triton-segmenter:py",
    [string]$ContainerName = "shadowgen-triton-segmenter",
    [int]$HttpPort = 8010,
    [int]$GrpcPort = 8011,
    [int]$MetricsPort = 8012,
    [int]$Resolution = 0,
    [string]$Device = "",
    [string]$ModelId = "ZhengPeng7/BiRefNet-matting",
    [string]$HfCacheDir = "",
    [switch]$NoBuild,
    [switch]$Gpu,
    [switch]$BindModelRepository,
    [switch]$Detach,
    [switch]$Wait,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host "Usage:"
    Write-Host "  tools\run_triton_segmenter_python.ps1 [-Gpu] [-BindModelRepository] [-Detach] [-Wait] [-NoBuild] [-HttpPort 8010] [-GrpcPort 8011] [-MetricsPort 8012] [-Resolution 512] [-HfCacheDir PATH]"
    Write-Host ""
    Write-Host "Starts a local Triton container for shadowgen_segmenter."
    Write-Host "Default host ports avoid FastAPI's local 8000:"
    Write-Host "  HTTP    http://127.0.0.1:8010"
    Write-Host "  gRPC    127.0.0.1:8011"
    Write-Host "  metrics http://127.0.0.1:8012/metrics"
    Write-Host ""
    Write-Host "By default the container starts without Docker GPU flags for dev bring-up."
    Write-Host "Use -Gpu when Docker Desktop NVIDIA GPU support is configured."
    Write-Host "Without -Gpu the helper defaults the model resolution to 512 to avoid CPU/OOM crashes in Docker Desktop."
    Write-Host "With -Gpu the helper defaults the model resolution to 1024 for quality."
    Write-Host "Use -Resolution to override either default."
    Write-Host "HuggingFace cache is mounted from the host to avoid filling Docker overlay storage."
    Write-Host "By default the model repository is baked into the image to avoid Windows bind-mount issues."
    Write-Host "Use -BindModelRepository only when Docker can mount this workspace path reliably."
    Write-Host "Use -Detach to run Triton in the background."
    Write-Host "Use -Wait with -Detach to wait until shadowgen_segmenter reports ready."
    exit 0
}

$dockerBin = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$tritonRoot = Resolve-Path (Join-Path $repoRoot "ops\triton")
$dockerfile = Join-Path $tritonRoot "Dockerfile.segmenter-python"
$modelRepository = Resolve-Path (Join-Path $tritonRoot "model_repository")

if ([string]::IsNullOrWhiteSpace($HfCacheDir)) {
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $HfCacheDir = Join-Path $env:LOCALAPPDATA "ShadowGen\triton-hf-cache"
    } else {
        $HfCacheDir = Join-Path $repoRoot "var\cache\huggingface-triton"
    }
}
$hfCachePath = New-Item -ItemType Directory -Path $HfCacheDir -Force
$resolvedHfCacheDir = (Resolve-Path $hfCachePath.FullName).Path

$effectiveResolution = $Resolution
if ($effectiveResolution -le 0) {
    $effectiveResolution = if ($Gpu) { 1024 } else { 512 }
}
$effectiveDevice = $Device
if ([string]::IsNullOrWhiteSpace($effectiveDevice)) {
    $effectiveDevice = if ($Gpu) { "cuda" } else { "cpu" }
}

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
    if ($LASTEXITCODE -ne 0) {
        throw "Docker image build failed for $ImageName."
    }
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
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $gpuProbeOutput = (& $dockerBin run --rm --gpus all $ImageName python3 -c "print('gpu runtime ok')" 2>&1)
        $gpuProbeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($gpuProbeExitCode -ne 0) {
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
Write-Host "  device:     $effectiveDevice"
Write-Host "  resolution: $effectiveResolution"
Write-Host "  HF cache:   $resolvedHfCacheDir -> /root/.cache/huggingface"
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
    "--shm-size", "2g",
    "-p", "${HttpPort}:8000",
    "-p", "${GrpcPort}:8001",
    "-p", "${MetricsPort}:8002",
    "-e", "HF_HOME=/root/.cache/huggingface",
    "-e", "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub",
    "-e", "SHADOWGEN_TRITON_SEGMENTER_MODEL_ID=$ModelId",
    "-e", "SHADOWGEN_TRITON_SEGMENTER_RESOLUTION=$effectiveResolution",
    "-e", "SHADOWGEN_TRITON_SEGMENTER_DEVICE=$effectiveDevice",
    "-v", "${resolvedHfCacheDir}:/root/.cache/huggingface"
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

if ($Detach) {
    $containerOutput = (& $dockerBin @runArgs 2>&1)
    $runExitCode = $LASTEXITCODE
    if ($runExitCode -ne 0) {
        throw "docker run failed for ${ContainerName}. Output: $containerOutput"
    }
    $containerId = ($containerOutput | Select-Object -Last 1)
    Write-Host "Triton container started: $containerId"
    Write-Host "Check readiness:"
    Write-Host "  .venv\Scripts\python.exe tools\check_triton_segmenter_ready.py http://127.0.0.1:$HttpPort --wait-seconds 240"
    Write-Host "Tail logs:"
    Write-Host "  docker logs -f $ContainerName"
    if ($Wait) {
        & (Join-Path $repoRoot ".venv\Scripts\python.exe") (Join-Path $repoRoot "tools\check_triton_segmenter_ready.py") "http://127.0.0.1:$HttpPort" "--wait-seconds" "240"
        if ($LASTEXITCODE -ne 0) {
            throw "Triton readiness check failed for http://127.0.0.1:$HttpPort."
        }
    }
} else {
    & $dockerBin @runArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker run failed for ${ContainerName}."
    }
}
