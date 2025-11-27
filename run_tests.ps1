Write-Host "run_tests.ps1 is deprecated. Use build.ps1 -Tests" -ForegroundColor Yellow
& "$PSScriptRoot\build.ps1" -Tests
