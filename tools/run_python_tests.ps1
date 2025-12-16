param(
    [string]$Python = "C:/Users/ireal/Documents/pycauset/.venv/Scripts/python.exe",
    [string]$Pattern = "tests/python/test_*.py"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PYTHONPATH = "$repoRoot\python;$repoRoot"

$tests = Get-ChildItem $Pattern | Sort-Object Name
$failed = New-Object System.Collections.Generic.List[string]

foreach ($t in $tests) {
    Write-Host ("\n==== RUN " + $t.Name + " ====")
    & $Python $t.FullName
    if ($LASTEXITCODE -ne 0) {
        $failed.Add($t.Name)
    }
}

if ($failed.Count -ne 0) {
    Write-Host "\nFAILED TEST FILES:"
    $failed | ForEach-Object { Write-Host " - $_" }
    exit 1
}

Write-Host "\nOK: all Python test scripts passed"
