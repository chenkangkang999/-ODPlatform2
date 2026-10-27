[CmdletBinding()]
param(
    [string]$Python = "",
    [ValidateRange(1024, 65535)][int]$Port = 8510,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$repoDir = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path $repoDir "apps/web-frontend"
$manageFile = Join-Path $repoDir "apps/web-backend/manage.py"
if (-not $Python) {
    $candidates = @(
        (Join-Path $repoDir ".venv/Scripts/python.exe"),
        (Join-Path (Split-Path -Parent $repoDir) "web-env/Scripts/python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { $Python = $candidate; break }
    }
    if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
}
if (-not $SkipBuild) {
    Push-Location $frontendDir
    try {
        if (-not (Test-Path -LiteralPath "node_modules")) {
            & npm.cmd ci
            if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
        }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }
    } finally { Pop-Location }
}
if (-not (Test-Path -LiteralPath (Join-Path $frontendDir "dist/index.html"))) {
    throw "Build the frontend first: cd apps/web-frontend; npm ci; npm run build"
}
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = Join-Path $repoDir "apps/platform/src"
Push-Location $repoDir
try {
    & $Python $manageFile check
    if ($LASTEXITCODE -ne 0) { throw "Django configuration check failed. Install apps/web-backend/requirements.txt first." }
    Write-Host "FishScope local preview: http://127.0.0.1:$Port/"
    & $Python $manageFile runserver "127.0.0.1:$Port" --noreload
} finally { Pop-Location }
