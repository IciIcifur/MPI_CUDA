param(
    [Parameter(Mandatory=$true)][ValidateSet("cpu","cuda")][string]$Target
)

function Get-RepoRoot {
    $scriptPath = $MyInvocation.MyCommand.Path
    if (-not $scriptPath) { $scriptPath = $PSCommandPath }
    return (Split-Path -Parent (Split-Path -Parent $scriptPath))
}

function Find-BinarySimple {
    param([string]$exeName, [string]$repoRoot)

    $buildDirs = Get-ChildItem -Path $repoRoot -Directory -Filter "cmake-build-*" -ErrorAction SilentlyContinue
    foreach ($d in $buildDirs) {
        $pDebug = Join-Path (Join-Path $d.FullName "Debug") $exeName
        if (Test-Path $pDebug) { return (Get-Item $pDebug).FullName }
        $pRoot = Join-Path $d.FullName $exeName
        if (Test-Path $pRoot) { return (Get-Item $pRoot).FullName }
    }

    try {
        $found = Get-ChildItem -Path $repoRoot -Filter $exeName -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { return $found.FullName }
    } catch {}

    return $null
}

$repoRoot = Get-RepoRoot
Write-Host "Repo root:" $repoRoot

$configPath = Join-Path $repoRoot "data\config_n_body.json"
if (-not (Test-Path $configPath)) {
    Write-Host ""
    Write-Host "No configuration file found."
    Write-Host "Please add JSON config at: data/config_n_body.json with fields:"
    Write-Host "  {"
    Write-Host "    \"tend\": <number>,"
    Write-Host "    \"dt\": <number>,"
    Write-Host "    \"input\": \"<input_filename_relative_to_data>\""
    Write-Host "  }"
    Write-Host ""
    Write-Host "Example: { \"tend\": 86400, \"dt\": 3600, \"input\": \"test_2.txt\" }"
    exit 1
}

try {
    $jsonText = Get-Content -Raw -Path $configPath
    $cfg = $jsonText | ConvertFrom-Json -ErrorAction Stop
} catch {
    Write-Error "Failed to parse JSON config: $configPath"
    exit 1
}

if (-not $cfg.tend -or -not $cfg.dt -or -not $cfg.input) {
    Write-Error "Config missing required fields. Ensure 'tend', 'dt' and 'input' are present in $configPath"
    exit 1
}

$inputValue = $cfg.input.ToString()
if ([System.IO.Path]::IsPathRooted($inputValue)) {
    $inputPath = $inputValue
} else {
    $inputPath = Join-Path (Join-Path $repoRoot "data") $inputValue
}

Write-Host "Using config:" $configPath
Write-Host ("tend = {0}, dt = {1}" -f $cfg.tend, $cfg.dt)
Write-Host "Input file (resolved):" $inputPath

if (-not (Test-Path $inputPath)) {
    Write-Error "Input file specified in config not found: $inputPath"
    Write-Host "Please create the input file (relative to data/) or correct the 'input' field in config."
    exit 1
}

if ($Target -eq "cpu") {
    $outDir = Join-Path $repoRoot "results\nbody_cpu"
    $exeName = "n_body_cpu.exe"
} else {
    $outDir = Join-Path $repoRoot "results\nbody_cuda"
    $exeName = "n_body_cuda.exe"
}
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outCsv = Join-Path $outDir "trajectories.csv"
Write-Host "Output will be:" $outCsv

Write-Host "Looking for binary:" $exeName
$binary = Find-BinarySimple -exeName $exeName -repoRoot $repoRoot
if (-not $binary) {
    Write-Error "Executable $exeName not found. Please build the project (use scripts/rebuild_task2.ps1)."
    exit 1
}
Write-Host "Found binary:" $binary

$ci = [System.Globalization.CultureInfo]::InvariantCulture
$tendStr = ([double]$cfg.tend).ToString($ci)
$dtStr = ([double]$cfg.dt).ToString($ci)

$args = @($tendStr, $inputPath, $dtStr, $outCsv)
Write-Host "Running:" "`"$binary`" $($args -join ' ')"
$proc = Start-Process -FilePath $binary -ArgumentList $args -NoNewWindow -Wait -PassThru
if ($proc) {
    Write-Host "Process exit code:" $proc.ExitCode
} else {
    Write-Error "Failed to start process."
    exit 1
}

if ($proc -and $proc.ExitCode -eq 0) {
    $plotScript = Join-Path $repoRoot "scripts\plot_orbits.py"
    if (Test-Path $plotScript) {
        Write-Host "Calling plotting script..."
        try {
            & python $plotScript --file $outCsv --no-show
            Write-Host "Plotting finished."
        } catch {
            Write-Host "Failed to run plotting script. Ensure python and required packages are installed."
        }
    } else {
        Write-Host "Plotting script not found at scripts\plot_orbits.py - skipping plotting."
    }
}

exit $proc.ExitCode