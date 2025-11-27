# Setup Instructions

## 1. Install a C++ Compiler
Your system is missing a C++ compiler. Since you are on Windows, the recommended way is to install **Visual Studio Build Tools**.

1.  Download **Visual Studio Community 2022** (Free) from: [https://visualstudio.microsoft.com/vs/community/](https://visualstudio.microsoft.com/vs/community/)
2.  Run the installer.
3.  **Crucial Step**: In the "Workloads" tab, check the box for **"Desktop development with C++"**.
    *   This ensures the MSVC compiler (`cl.exe`) and Windows SDK are installed.
4.  Click **Install**.

## 2. Verify Installation
After installation completes:
1.  **Restart Visual Studio Code** (completely close and reopen it) to refresh environment variables.
2.  Open a terminal and try running:
    ```powershell
    cl
    ```
    *Note: `cl` might not be in your global PATH, which is fine. CMake usually finds it automatically.*

## 3. Build & Test
Use the consolidated build script:
```powershell
# Build everything and run C++ tests + Python module
.\build.ps1 -All

# Only run C++ tests
.\build.ps1 -Tests

# Only build the Python module
.\build.ps1 -Python
```
