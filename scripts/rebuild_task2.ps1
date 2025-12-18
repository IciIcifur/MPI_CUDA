param(
    [ValidateSet("cpu","cuda")][string]$Target = ""
)

function Get-RepoRoot {
    $scriptPath = $MyInvocation.MyCommand.Path
    if (-not $scriptPath) { $scriptPath = $PSCommandPath }
    return (Split-Path -Parent (Split-Path -Parent $scriptPath))
}

$repoRoot = Get-RepoRoot
Write-Host "Repo root:" $repoRoot

$task2Dir = Join-Path $repoRoot "task2"
if (Test-Path $task2Dir -PathType Container) {
    $sourceDir = $task2Dir
    Write-Host "Using 'task2' subfolder as CMake source: $sourceDir"
} else {
    $sourceDir = $repoRoot
    Write-Host "Using repo root as CMake source: $sourceDir"
}

$buildDir = Join-Path $repoRoot "cmake-build-task2-Debug"

if (Test-Path $buildDir) {
    Write-Host "Removing previous build directory: $buildDir"
    try {
        Remove-Item -LiteralPath $buildDir -Recurse -Force -ErrorAction Stop
    } catch {
        Write-Error "Failed to remove build directory: $_.Exception.Message"
        exit 1
    }
}

$targetsToBuild = @()
if ($Target -eq "cpu") { $targetsToBuild = @("cpu") }
elseif ($Target -eq "cuda") { $targetsToBuild = @("cuda") }
else { $targetsToBuild = @("cpu","cuda") }  # default: build both

foreach ($t in $targetsToBuild) {
    Write-Host "---- Building target: $t ----"

    $cmakeArgs = @("-S", $sourceDir, "-B", $buildDir, "-A", "x64", "-DCMAKE_BUILD_TYPE=Debug")
    if ($t -eq "cpu") {
        $cmakeArgs += "-DBUILD_CPU=ON"
        $cmakeArgs += "-DBUILD_CUDA=OFF"
    } else {
        $cmakeArgs += "-DBUILD_CPU=OFF"
        $cmakeArgs += "-DBUILD_CUDA=ON"
    }

    Write-Host "Configuring with CMake: cmake $($cmakeArgs -join ' ')"
    $proc = Start-Process -FilePath "cmake" -ArgumentList $cmakeArgs -NoNewWindow -Wait -PassThru
    if (-not $proc -or $proc.ExitCode -ne 0) {
        Write-Error "CMake configure failed for target '$t' (exit code $($proc.ExitCode))."
        exit 1
    }

    $buildArgs = @("--build", $buildDir, "--config", "Debug", "--parallel", "8")
    Write-Host "Building: cmake $($buildArgs -join ' ')"
    $proc2 = Start-Process -FilePath "cmake" -ArgumentList $buildArgs -NoNewWindow -Wait -PassThru
    if (-not $proc2 -or $proc2.ExitCode -ne 0) {
        Write-Error "Build failed for target '$t' (exit code $($proc2.ExitCode))."
        exit 1
    }

    Write-Host "Build for '$t' finished successfully."
}

Write-Host "All requested builds finished. Build directory: $buildDir"
exit 0