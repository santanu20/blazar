# pallama bootstrap installer - Windows (PowerShell 5.1+ / pwsh).
#
#   irm <raw-url-of-this-file> | iex
#
# Verifies the asset sha256 from the GitHub release API (the same source of
# truth as `pallama engine update`) before installing anything. No admin:
# installs to LOCALAPPDATA\Programs\pallama.
#
# Parameters / env overrides:
#   -Version            pin a release tag (e.g. v0.1.0)   [env: PALLAMA_VERSION]
#   -Repo               GitHub owner/name                 [env: PALLAMA_REPO]
#   -InstallDir         binary destination
#   [env] PALLAMA_INSTALL_BASE_URL  replace the GitHub API base (mirrors, tests)
#   [env] GITHUB_TOKEN              optional API token

[CmdletBinding()]
param(
    [string]$Version = $env:PALLAMA_VERSION,
    [string]$Repo = $env:PALLAMA_REPO,
    [string]$InstallDir,
    [string]$ApiBase = $env:PALLAMA_INSTALL_BASE_URL
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Fail([string]$Message) { Write-Host "ERROR: $Message" -ForegroundColor Red; exit 1 }

if (-not $Repo -and -not $ApiBase) {
    Fail 'PALLAMA_REPO is not configured. Run with -Repo owner/pallama (or set $env:PALLAMA_REPO).'
}
if (-not $ApiBase) { $ApiBase = "https://api.github.com/repos/$Repo" }

$Target = 'x86_64-pc-windows-msvc'

$headers = @{ 'User-Agent' = 'pallama-install' }
if ($env:GITHUB_TOKEN) { $headers['Authorization'] = "Bearer $($env:GITHUB_TOKEN)" }

$releasePath = if ($Version) { "releases/tags/$Version" } else { 'releases/latest' }
Write-Host ">>> Fetching release metadata from $ApiBase/$releasePath"
try {
    $release = Invoke-RestMethod -Uri "$ApiBase/$releasePath" -Headers $headers
} catch {
    Fail "could not fetch release metadata: $($_.Exception.Message)"
}

$assetName = "pallama-$($release.tag_name)-$Target.zip"
$asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
if (-not $asset) {
    $available = ($release.assets | ForEach-Object { $_.name }) -join ' '
    Fail "asset $assetName not found in release $($release.tag_name). Available: $available"
}
if (-not $asset.digest -or $asset.digest -notlike 'sha256:*') {
    Fail "release metadata has no sha256 digest for $assetName yet - GitHub computes it shortly after upload; retry in a minute. Refusing unverified install."
}
$expected = $asset.digest -replace '^sha256:', ''

if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA 'Programs\pallama' }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("pallama-install-" + [GUID]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    Write-Host ">>> Downloading $assetName ($($release.tag_name))..."
    $zip = Join-Path $tmp $assetName
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -Headers $headers

    $got = (Get-FileHash -Path $zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($got -ne $expected) {
        Fail "sha256 mismatch for ${assetName}: expected $expected, got $got - download corrupted or tampered; not installing"
    }
    Write-Host '>>> sha256 verified'

    $extract = Join-Path $tmp 'extract'
    Expand-Archive -Path $zip -DestinationPath $extract
    $exe = Join-Path $extract 'pallama.exe'
    if (-not (Test-Path $exe)) { Fail 'archive did not contain a pallama.exe at its root' }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    # Stop a running daemon so the exe file is not locked.
    Get-Process -Name pallama -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Copy-Item $exe (Join-Path $InstallDir 'pallama.exe') -Force

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { $userPath = '' }
    if ($userPath -notlike "*$InstallDir*") {
        [Environment]::SetEnvironmentVariable('Path', "$userPath;$InstallDir", 'User')
        Write-Host ">>> NOTE: added $InstallDir to your user PATH - open a new terminal for it to take effect"
    }

    $installedExe = Join-Path $InstallDir 'pallama.exe'
    $ver = & $installedExe --version
    Write-Host ">>> Installed pallama $ver to $installedExe"
    Write-Host '>>> Next: pallama engine update && pallama pull <model> && pallama run <model>'
    Write-Host '>>> All inference is upstream llama.cpp - ggml, ggerganov and contributors did the hard parts.'
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
