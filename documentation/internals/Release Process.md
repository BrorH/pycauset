# Release Process

PyCauset uses an automated release pipeline based on GitHub Actions. This document outlines how versioning and publishing to PyPI are handled.

## Automated Releases

The primary way to release a new version is by pushing to the `main` branch.

1.  **Commit your changes**: Ensure your work is committed.
2.  **Push to main**:
    ```bash
    git push origin main
    ```
3.  **Workflow Trigger**: The GitHub Action `Publish to PyPI` will start automatically.
    *   **Bump Version**: It calculates the next **patch** version (e.g., `0.2.4` -> `0.2.5`).
    *   **Tag**: It creates a new git tag and a GitHub Release using your commit message as the notes.
    *   **Build & Publish**: It builds wheels for Windows, macOS, and Linux, and publishes them to PyPI.

## Manual Releases

If you need to bump a minor/major version or set a specific version number, you can use the helper script or git tags directly.

### Using the Helper Script (`release.ps1`)

A PowerShell script is provided in the root directory to simplify manual tagging.

**Bump Minor Version:**
```powershell
.\release.ps1 -Type minor
```
*Example: `0.2.4` -> `0.3.0`*

**Bump Major Version:**
```powershell
.\release.ps1 -Type major
```
*Example: `0.2.4` -> `1.0.0`*

**Set Specific Version:**
```powershell
.\release.ps1 -SetVersion 0.4.0
```
*Example: Jumps directly to `0.4.0`*

### Using Git Tags

You can also manually create and push a tag. The CI/CD pipeline will detect the new tag and build/publish that specific version (skipping the auto-bump step).

```bash
git tag v0.4.0
git push origin v0.4.0
```

## CI/CD Workflow Details

The workflow is defined in `.github/workflows/publish.yml`.

*   **Triggers**:
    *   `push` to `main`: Triggers auto-bump (patch) and release.
    *   `push` of tags (`v*`): Triggers build and release for that tag.
    *   `workflow_dispatch`: Allows manual triggering from GitHub Actions UI.
*   **Versioning**: Uses `setuptools_scm` to determine the package version from git tags.
*   **Build System**: Uses `cibuildwheel` to build binary wheels for multiple platforms.
*   **Publishing**: Uses PyPI Trusted Publishing (OIDC) to upload artifacts.
