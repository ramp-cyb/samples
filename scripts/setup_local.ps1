# Set up homesim in a local virtualenv and verify it works (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1 -Train
param([switch]$Train)
$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) { throw "Python not found. Install Python 3.10 or newer from python.org." }

$version = & $py.Source -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "==> creating virtualenv in .venv (Python $version)"
& $py.Source -m venv .venv
$venvPy = ".venv\Scripts\python.exe"

Write-Host "==> installing homesim"
& $venvPy -m pip install --quiet --upgrade pip
if ($Train) { & $venvPy -m pip install -e ".[test,train]" } else { & $venvPy -m pip install -e ".[test]" }

Write-Host "==> running the test suite"
& $venvPy -m pytest -q

Write-Host "==> generating a small sample dataset"
& $venvPy scripts\generate_data.py --train-houses 20 --per-house 4 --eval-houses 8 --eval-per-house 2

Write-Host "==> checking the oracle scores perfectly on it"
& $venvPy scripts\run_eval.py --policy oracle --data data\eval_in_dist.jsonl

Write-Host ""
Write-Host "Setup complete."
Write-Host "  Activate:     .venv\Scripts\activate"
Write-Host "  Live UI:      python scripts\serve_ui.py   then open http://127.0.0.1:8000"
Write-Host "  Full dataset: python scripts\generate_data.py --train-houses 400 --per-house 6"
