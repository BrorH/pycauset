# Release Process

This page describes how PyCauset is released and versioned.

## Automated releases

The primary way to release a new version is by pushing to the `main` branch.

1. **Commit your changes**: ensure your work is committed.
2. **Push to main**:
    ```bash
    git push origin main
    ```
3. **Workflow trigger**: the GitHub Action `Publish to PyPI` will start automatically.
    - **Bump version**: calculates the next **patch** version (e.g., `0.2.4` -> `0.2.5`).
    - **Tag**: creates a new git tag and a GitHub Release using your commit message as the notes.
    - **Build & publish**: builds wheels for Windows, macOS, and Linux, and publishes them to PyPI.

## Manual releases

If you need to bump a minor/major version or set a specific version number, you can use the helper script or git tags directly.

### Using the helper script (`release.ps1`)

Bump minor version:
```powershell
.\release.ps1 -Type minor
```

Bump major version:
```powershell
.\release.ps1 -Type major
```

Set specific version:
```powershell
.\release.ps1 -SetVersion 0.4.0
```

### Using git tags

```bash
git tag v0.4.0
git push origin v0.4.0
```

## CI/CD workflow details

The workflow is defined in `.github/workflows/publish.yml`.

- Triggers:
  - `push` to `main`: triggers auto-bump (patch) and release
  - `push` of tags (`v*`): triggers build and release for that tag
  - `workflow_dispatch`: manual trigger from GitHub Actions UI
- Versioning: uses `setuptools_scm` to determine the package version from git tags
- Build system: uses `cibuildwheel` to build binary wheels for multiple platforms
- Publishing: uses PyPI Trusted Publishing (OIDC) to upload artifacts
