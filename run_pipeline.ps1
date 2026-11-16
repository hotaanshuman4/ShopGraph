# ==============================================================================
# ShopGraph: Master PowerShell Pipeline Execution Script
# ==============================================================================
# Usage:
#   .\run_pipeline.ps1
# ==============================================================================

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  SHOPGRAPH: DISTRIBUTED E-COMMERCE ANALYTICS & RECOMMENDATION SYSTEM" -ForegroundColor Yellow
Write-Host "  Dataset: UCI Online Retail II (~1.07 Million Invoices, 2009-2011)" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = "$ScriptDir\..\python\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python.exe"
}

Write-Host "Using Python Runtime: $PythonExe" -ForegroundColor Green
Write-Host "Starting complete end-to-end execution..." -ForegroundColor Green

& $PythonExe "$ScriptDir\run_all.py"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  SHOPGRAPH PIPELINE EXECUTION COMPLETED!" -ForegroundColor Green
Write-Host "  Results and presentation figures are available in: outputs/" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan
