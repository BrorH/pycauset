$ErrorActionPreference = "Stop"
$root = "C:\Users\ireal\Documents\Projects\pycauset"
$work = "$root\third_party\openblas_src"
$ver = "0.3.28"
$zip = "$root\third_party\openblas-$ver.zip"

New-Item -ItemType Directory -Force -Path "$root\third_party" | Out-Null

# Download OpenBLAS source if not present
if (-not (Test-Path $zip)) {
    Write-Host "Downloading OpenBLAS $ver source..."
    Invoke-WebRequest -Uri "https://github.com/OpenMathLib/OpenBLAS/archive/v$ver.zip" -OutFile $zip -UseBasicParsing
}

if (-not (Test-Path "$work\Makefile")) {
    Write-Host "Extracting..."
    Expand-Archive -Path $zip -DestinationPath "$root\third_party" -Force
    # Expand-Archive creates OpenBLAS-0.3.28; rename to openblas_src
    $extracted = "$root\third_party\OpenBLAS-$ver"
    if (Test-Path $extracted) {
        if (Test-Path $work) { Remove-Item $work -Recurse -Force }
        Rename-Item $extracted $work
    }
}

# Build with MinGW: DYNAMIC_ARCH (Haswell/Skylake/Zen), threaded, LP64.
# Combine WinLibs (gcc/gfortran/make/binutils) with Git's POSIX utils (sh/uname/sed).
Write-Host "Building OpenBLAS (DYNAMIC_ARCH=1 USE_THREAD=1)... this takes 10-20 min."
Set-Location $work
$winlibs = "C:\Users\ireal\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
$gitposix = "C:\Program Files\Git\usr\bin"
$env:PATH = "$winlibs;$gitposix;$env:PATH"

# Use `make` if present, else mingw32-make
$makeExe = "mingw32-make"
$makeArgs = @("DYNAMIC_ARCH=1","USE_THREAD=1","NUM_THREADS=24","BINARY=64","NO_SHARED=0","NO_STATIC=1","NO_LAPACKE=0","NO_CBLAS=0","CC=gcc","FC=gfortran","HOSTCC=gcc")
# Run via cmd /c so PowerShell does not treat mingw32-make's stderr chatter
# (e.g. "ar: creating ...") as a terminating error.
$argStr = ($makeArgs -join " ")
cmd /c "`"$winlibs\$makeExe`" $argStr > `"$root\openblas_build.log`" 2>&1"
Write-Host "make exit code: $LASTEXITCODE"

Write-Host "Build complete. Output files:"
Get-ChildItem "$work\libopenblas*.dll","$work\*.dll","$work\exports\*.def" -ErrorAction SilentlyContinue | Select-Object FullName, Length | Out-String | Write-Host

# Generate an MSVC-compatible import lib
$dll = Get-ChildItem "$work\libopenblas.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($dll) {
    Write-Host "Generating import lib from $($dll.FullName)..."
    & gendef $dll.FullName 2>&1 | Out-Host
    $def = "$work\libopenblas.def"
    if (Test-Path $def) {
        & dlltool -d $def -l "$work\libopenblas_msvc.lib" -D libopenblas.dll 2>&1 | Out-Host
        Write-Host "Import lib: $work\libopenblas_msvc.lib"
    }
}
Write-Host "DONE"
