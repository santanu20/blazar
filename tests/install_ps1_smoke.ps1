# Windows installer smoke test (CI, windows-latest). It drives
# scripts/install.ps1 the way a user does and asserts the visible outcome of
# each lane: a plain install, the uninstall, and the start-at-logon
# scheduled task that -WithService registers.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests/install_ps1_smoke.ps1 -Version v0.34.3
#
# The installer under test always runs in a child Windows PowerShell, so this
# script's own strict-mode state cannot mask an installer defect. The child
# gets an empty line on stdin: an unexpected prompt then takes its default
# instead of hanging the job.
param(
    [string]$Version = 'v0.34.3'
)

$ErrorActionPreference = 'Stop'
$installer = Join-Path $PSScriptRoot '..\scripts\install.ps1'
$installDir = Join-Path $env:RUNNER_TEMP 'blazar-ci'
# The smoke covers the installer, not the engine download: the bootstrap
# lane is the largest download of an install and is exercised elsewhere.
$env:BLAZAR_INSTALL_ENGINE = '0'
$exe = Join-Path $installDir 'blazar.exe'
$taskName = 'blazar'

function Assert-That {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw "windows installer smoke: $Message" }
    Write-Host ">>> ok - $Message"
}

# The user PATH is a REG_EXPAND_SZ value: read it unexpanded so a comparison
# against the install dir is exact.
function Get-UserPathEntries {
    # The 4-argument static GetValue is .NET 5+ only; the key overload also
    # runs under Windows PowerShell 5.1.
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment')
    try {
        $raw = $key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    } finally {
        $key.Close()
    }
    return $raw -split ';'
}

function Invoke-Installer {
    param([string[]]$InstallerArgs)
    # Scope-local: a child's stderr lines must not turn into terminating
    # errors under the caller's preference.
    $ErrorActionPreference = 'Continue'
    '' | & powershell -NoProfile -ExecutionPolicy Bypass -File $installer @InstallerArgs
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "windows installer smoke: installer exited $code ($InstallerArgs)" }
}

function Get-VersionOutput {
    $ErrorActionPreference = 'Continue'
    return (& $exe --version 2>&1 | Out-String)
}

function Get-BlazarTask {
    return Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}

function Remove-InstallDir {
    if (Test-Path $installDir) { Remove-Item $installDir -Recurse -Force }
}

Remove-InstallDir

# 1. plain install: binary on disk, a working executable, one PATH entry,
#    and no background registration.
Invoke-Installer @('-InstallDir', $installDir, '-Version', $Version)
Assert-That (Test-Path $exe) 'plain install places blazar.exe in the install dir'
$versionText = Get-VersionOutput
Assert-That ($versionText.Trim().Length -gt 0) 'blazar.exe --version prints a version'
Assert-That (@(Get-UserPathEntries | Where-Object { $_ -eq $installDir }).Count -eq 1) 'plain install adds the install dir to the user PATH once'
Assert-That ($null -eq (Get-BlazarTask)) 'plain install registers no scheduled task'

# 2. plain uninstall: binary and PATH entry gone.
Invoke-Installer @('-Uninstall', '-InstallDir', $installDir)
Assert-That (-not (Test-Path $exe)) 'uninstall removes blazar.exe'
Assert-That (@(Get-UserPathEntries | Where-Object { $_ -eq $installDir }).Count -eq 0) 'uninstall removes the PATH entry'

# 3. -WithService: the logon task exists after install and is gone after
#    uninstall.
Invoke-Installer @('-InstallDir', $installDir, '-Version', $Version, '-WithService')
Assert-That ($null -ne (Get-BlazarTask)) '-WithService registers the logon task'
Invoke-Installer @('-Uninstall', '-InstallDir', $installDir)
Assert-That ($null -eq (Get-BlazarTask)) 'uninstall unregisters the logon task'

Remove-InstallDir
Write-Host 'windows installer smoke: ok'
