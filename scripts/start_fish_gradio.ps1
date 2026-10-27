[CmdletBinding()]
param([string]$Python = "", [ValidateRange(1024, 65535)][int]$Port = 7860)
$ErrorActionPreference = "Stop"
$repoDir = Split-Path -Parent $PSScriptRoot
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
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = Join-Path $repoDir "apps/platform/src"
Push-Location $repoDir
try {
    & $Python (Join-Path $repoDir "apps/web-backend/run_gradio.py") --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Gradio launch failed; check optional UI dependencies and local fish weights." }
} finally { Pop-Location }
