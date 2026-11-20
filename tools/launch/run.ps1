# Run HCPD with the project virtual environment (recommended)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment not found. Run: python -m venv .venv"
    exit 1
}

& $Python (Join-Path $Root "main.py") @args
