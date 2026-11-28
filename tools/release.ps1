param(
    [string]$Type = "patch", # patch, minor, major
    [string]$SetVersion = ""
)

# Tags a release version and pushes the tag. A tag publishes nothing by itself: the release
# workflow (publish.yml) runs on a published GitHub Release, which the last step creates (see
# documentation/project/Release Process.md). Every git call is checked (review 3.15 C-17: a
# failed tag or push still ended in "Done!").
Set-Location (Join-Path $PSScriptRoot '..')
function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) { Write-Error "git $($args -join ' ') failed ($LASTEXITCODE)"; exit 1 }
}

# Release from main, equal to origin/main
Invoke-Git fetch origin main --tags
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "main") { Write-Error "Release from main (on '$branch')."; exit 1 }
if ((git rev-parse HEAD).Trim() -ne (git rev-parse origin/main).Trim()) {
    Write-Error "main differs from origin/main: pull or push first."; exit 1
}

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
    # Auto-increment from the highest version tag (git describe gave the nearest tag
    # reachable from HEAD, not the highest version).
    $latestTag = (git tag --list 'v*' --sort=-v:refname | Select-Object -First 1)
    if (-not $latestTag) {
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
$confirmation = Read-Host "Create and push tag $newTag? (y/n)"
if ($confirmation -ne 'y') {
    Write-Host "Aborted."
    exit 0
}

# Execute
Write-Host "Tagging $newTag..."
Invoke-Git tag $newTag
Write-Host "Pushing..."
Invoke-Git push origin $newTag

Write-Host "Tag $newTag pushed. A tag publishes nothing by itself; create the GitHub Release to run the publish workflow:"
Write-Host "  gh release create $newTag --generate-notes"
