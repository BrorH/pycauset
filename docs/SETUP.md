# Setup Instructions

## 1. Install a C++ Compiler
Install **Visual Studio Community 2022** with the **Desktop development with C++** workload (or another MSVC-compatible toolchain) so `cl.exe` and the Windows SDK are available.

## 2. Verify Installation
Restart VS Code, open a terminal, and run:
```powershell
cl
```
Seeing the MSVC version output confirms the compiler is discoverable.

## 3. Build & Test
Use the consolidated build script:
```powershell
# Build everything and run C++ tests + Python module
./build.ps1 -All

# Only run C++ tests
./build.ps1 -Tests

# Only build the Python module
./build.ps1 -Python
```
