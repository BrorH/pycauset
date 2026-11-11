param(
    [string]$Type = "patch", # patch, minor, major
    [string]$SetVersion = ""
)

# Ensure git status is clean
$status = git status --porcelain
if ($status) {
    Write-Error "Working directory is not clean. Please commit or stash changes first."
    exit 1
}

if ($SetVersion) {
    # Use provided version
    if ($SetVersion -notmatch "^v?\d+\.\d+\.\d+$") {
        Write-Error "Version must be in format vX.Y.Z or X.Y.Z (e.g. 0.4.0)"
        exit 1
    }
    if ($SetVersion -notmatch "^v") { $SetVersion = "v$SetVersion" }
    $newTag = $SetVersion
}
else {
    # Auto-increment logic
    git fetch --tags
    try {
        $latestTag = git describe --tags --abbrev=0 2>$null
        if (-not $latestTag) { throw "No tags found" }
    }
    catch {
        $latestTag = "v0.0.0"
        Write-Warning "No tags found, starting from v0.0.0"
    }

    if ($latestTag -match "^v?(\d+)\.(\d+)\.(\d+)$") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        $patch = [int]$matches[3]
    }
    else {
        Write-Error "Could not parse version from tag: $latestTag"
        exit 1
    }

    switch ($Type) {
        "major" { $major++; $minor = 0; $patch = 0 }
        "minor" { $minor++; $patch = 0 }
        "patch" { $patch++ }
        default { Write-Error "Invalid type. Use patch, minor, or major."; exit 1 }
    }
    $newTag = "v$major.$minor.$patch"
}

# Confirm
$confirmation = Read-Host "Create tag $newTag and publish it to PyPI (~270 MB of the 10 GB quota)? (y/n)"
if ($confirmation -ne 'y') {
    Write-Host "Aborted."
    exit 0
}

# Execute
Write-Host "Tagging $newTag..."
git tag $newTag
Write-Host "Pushing tag..."
git push origin $newTag

# Publishing a GitHub Release is what triggers the PyPI upload
# (.github/workflows/publish.yml). A bare tag push uploads nothing.
if (Get-Command gh -ErrorAction SilentlyContinue) {
    Write-Host "Creating GitHub Release $newTag (this starts the PyPI upload)..."
    gh release create $newTag --title $newTag --generate-notes
    Write-Host "Done! Follow the 'Publish to PyPI.' run in the Actions tab."
}
else {
    Write-Host "Tag pushed. To upload to PyPI, publish a GitHub Release for $newTag:"
    Write-Host "  https://github.com/BrorH/pycauset/releases/new?tag=$newTag"
    Write-Host "(Mark it as a pre-release instead if you only want to test the build.)"
}
