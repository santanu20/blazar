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
#   -WithService        ALSO register a start-at-logon scheduled task
#                       ("pallama") running `pallama serve` (console
#                       binaries cannot be NT services without a shim;
#                       a scheduled task is the dependency-free lane)
#   -Build              compile from source instead of downloading a
#                       release asset. Run it from a checkout, or pass
#                       -Repo to clone. Installs rustup via winget when
#                       cargo is missing (needs VS Build Tools C++ for
#                       linking; error message points there on failure).
#   -Uninstall          remove binary + PATH entry + task
#   ARM64 Windows hosts pick the aarch64 asset automatically and fall
#   back to the x86_64 one (emulated on Win11 ARM64) with a warning
#   when a release has no native build.
#   [env] PALLAMA_INSTALL_BASE_URL  replace the GitHub API base (mirrors, tests)
#   [env] PALLAMA_AUTO_DRIVER        '0' skips the GPU driver preflight advisory
#   [env] GITHUB_TOKEN               optional API token
#
#   flags need a saved script: irm <url> -OutFile install.ps1; .\install.ps1 -Build

[CmdletBinding()]
param(
    [string]$Version = $env:PALLAMA_VERSION,
    [string]$Repo = $env:PALLAMA_REPO,
    [string]$InstallDir,
    [string]$ApiBase = $env:PALLAMA_INSTALL_BASE_URL,
    [switch]$WithService,
    [switch]$Build,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Fail([string]$Message) { Write-Host "ERROR: $Message" -ForegroundColor Red; exit 1 }

$TaskName = 'pallama'

function Stop-PallamaGraceful([string]$ExePath) {
    # F155: drain in-flight requests first (`pallama stop` HTTP), then
    # force-kill only survivors — install.sh:170 parity.
    try {
        if ($ExePath -and (Test-Path $ExePath)) {
            & $ExePath stop *> $null
            Start-Sleep -Milliseconds 800
        }
    } catch { }
    Get-Process -Name pallama -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

if ($Uninstall) {
    if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA 'Programs\pallama' }
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host ">>> removed scheduled task $TaskName"
    }
    # F155: graceful drain before the hard kill.
    Stop-PallamaGraceful (Join-Path $InstallDir 'pallama.exe')
    if (Test-Path (Join-Path $InstallDir 'pallama.exe')) {
        Remove-Item $InstallDir -Recurse -Force
        Write-Host ">>> removed $InstallDir"
    }
    # F155b: exact-entry PATH removal (split on ';', drop the precise
    # install dir, rejoin) — substring matching over-matches sibling
    # directories and the leading-';'-only replace left stale entries.
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath) {
        $entries = @($userPath.Split(';') | Where-Object { $_ -ne '' -and $_ -ne $InstallDir })
        if ($entries.Count -ne @($userPath.Split(';') | Where-Object { $_ -ne '' }).Count) {
            [Environment]::SetEnvironmentVariable('Path', ($entries -join ';'), 'User')
            Write-Host '>>> removed PATH entry'
        }
    }
    Write-Host '>>> uninstall complete (models + config under LOCALAPPDATA are user data; delete manually if wanted)'
    exit 0
}

function Register-PallamaTask([string]$ExePath) {
    # Start at logon (user scope, no admin). The task restarts the
    # daemon only on CRASH (non-zero exit); `pallama stop` exits 0 and
    # is never undone. F156: restart settings explicit, not defaulted.
    $action = New-ScheduledTaskAction -Execute $ExePath -Argument 'serve'
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 3 -RestartInterval ([TimeSpan]::FromSeconds(30))
    Register-ScheduledTask -TaskName $script:TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $script:TaskName
    Write-Host ">>> registered + started scheduled task '$script:TaskName' ($ExePath serve)"
}

if (-not $Build -and -not $Repo -and -not $ApiBase) {
    Fail 'PALLAMA_REPO is not configured. Run with -Repo owner/pallama (or set $env:PALLAMA_REPO), or -Build from a checkout.'
}
if (-not $ApiBase) { $ApiBase = "https://api.github.com/repos/$Repo" }

$isArm64Host = $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64'
$Target = if ($isArm64Host) { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }

$headers = @{ 'User-Agent' = 'pallama-install' }
if ($env:GITHUB_TOKEN) { $headers['Authorization'] = "Bearer $($env:GITHUB_TOKEN)" }

if (-not $Build) {
    $releasePath = if ($Version) { "releases/tags/$Version" } else { 'releases/latest' }
    Write-Host ">>> Fetching release metadata from $ApiBase/$releasePath"
    try {
        $release = Invoke-RestMethod -Uri "$ApiBase/$releasePath" -Headers $headers
    } catch {
        # F155b: surface the API's own message (rate-limit etc.), not a
        # raw exception blob.
        $detail = if ($_.ErrorDetails.Message) { $_.ErrorDetails.Message } else { $_.Exception.Message }
        if ($detail -match 'rate limit') {
            Fail "GitHub API rate limit exceeded - set GITHUB_TOKEN and retry, or wait for the window to reset."
        }
        Fail "could not fetch release metadata: $detail"
    }

    $assetName = "pallama-$($release.tag_name)-$Target.zip"
    $asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
    if (-not $asset -and $isArm64Host) {
        Write-Host ">>> WARN: release $($release.tag_name) has no native ARM64 asset - falling back to x86_64 (runs emulated on Windows 11 ARM64)"
        $Target = 'x86_64-pc-windows-msvc'
        $assetName = "pallama-$($release.tag_name)-$Target.zip"
        $asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
    }
    if (-not $asset) {
        $available = ($release.assets | ForEach-Object { $_.name }) -join ' '
        Fail "asset $assetName not found in release $($release.tag_name). Available: $available"
    }
    if (-not $asset.digest -or $asset.digest -notlike 'sha256:*') {
        Fail "release metadata has no sha256 digest for $assetName yet - GitHub computes it shortly after upload; retry in a minute. Refusing unverified install."
    }
    $expected = $asset.digest -replace '^sha256:', ''
}

if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA 'Programs\pallama' }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("pallama-install-" + [GUID]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    if ($Build) {
        if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
            Write-Host '>>> cargo not found - installing rustup via winget (Rustlang.Rustup)...'
            if (Get-Command winget -ErrorAction SilentlyContinue) {
                winget install --id Rustlang.Rustup -e --silent --accept-source-agreements --accept-package-agreements
                $env:Path = "$env:USERPROFILE/.cargo/bin$([IO.Path]::PathSeparator)$env:Path"
            }
            if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
                Fail '-Build needs Rust (MSVC): winget install Rustlang.Rustup (+ VS Build Tools C++ workload for linking), open a NEW terminal, retry.'
            }
        }
        $srcRoot = $null
        if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot '../Cargo.toml'))) {
            $srcRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
            Write-Host ">>> building from checkout $srcRoot"
        } else {
            if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
                Fail '-Build outside a checkout clones via git, but git is missing. Install it: winget install Git.Git'
            }
            if (-not $Repo) {
                Fail '-Build outside a checkout needs -Repo owner/name (or set $env:PALLAMA_REPO) to clone the sources.'
            }
            $srcRoot = Join-Path $tmp 'src'
            $cloneUrl = "https://github.com/$Repo.git"
            Write-Host ">>> cloning $cloneUrl$(if ($Version) { " (tag $Version)" })..."
            if ($Version) { & git clone --depth 1 --branch $Version $cloneUrl $srcRoot }
            else { & git clone --depth 1 $cloneUrl $srcRoot }
            if ($LASTEXITCODE -ne 0) { Fail "git clone failed: $cloneUrl" }
        }
        Write-Host '>>> compiling (cargo build --release; first build takes a few minutes)...'
        & cargo build --release --manifest-path (Join-Path $srcRoot 'Cargo.toml')
        if ($LASTEXITCODE -ne 0) {
            Fail 'cargo build failed (output above). If it mentions link.exe: winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
        }
        $exe = Join-Path $srcRoot 'target/release/pallama.exe'
        if (-not (Test-Path $exe)) { Fail 'cargo build reported success but target\release\pallama.exe was not produced' }
    } else {
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
    }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    # Stop a running daemon so the exe file is not locked (F155: HTTP
    # drain first, force only for survivors).
    $taskExisted = [bool](Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)
    Stop-PallamaGraceful (Join-Path $InstallDir 'pallama.exe')
    Copy-Item $exe (Join-Path $InstallDir 'pallama.exe') -Force

    # F155b: exact-entry PATH add — `-notlike "*dir*"` over-matched
    # sibling directories (Programs\pallama vs Programs\pallama-old).
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { $userPath = '' }
    $entries = @($userPath.Split(';') | Where-Object { $_ -ne '' })
    if ($entries -notcontains $InstallDir) {
        [Environment]::SetEnvironmentVariable('Path', (($entries + $InstallDir) -join ';'), 'User')
        Write-Host ">>> NOTE: added $InstallDir to your user PATH - open a new terminal for it to take effect"
    }

    $installedExe = Join-Path $InstallDir 'pallama.exe'
    $ver = & $installedExe --version
    Write-Host ">>> Installed pallama $ver to $installedExe"
    Write-Host ">>> Next: pallama pull <model> (find: pallama search qwen3) | pallama run <model> | pallama doctor"
    # GPU preflight (advise-only; Windows drivers come from the vendor's
    # own installer — this script never installs kernel drivers). The
    # engine bootstrap below picks its asset by driver presence, so a
    # driverless NVIDIA box would silently serve on CPU: detect and say
    # so instead. Opt out: PALLAMA_AUTO_DRIVER=0.
    if ($env:PALLAMA_INSTALL_ENGINE -ne '0' -and $env:PALLAMA_AUTO_DRIVER -ne '0') {
        try { $gpus = @(Get-CimInstance Win32_VideoController -ErrorAction Stop) } catch { $gpus = @() }
        if ($gpus.Count -gt 0) {
            $nvidia = $gpus | Where-Object { $_.Name -match 'NVIDIA' }
            if ($nvidia -and -not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
                Write-Host '>>> WARN: NVIDIA GPU detected but no NVIDIA driver (nvidia-smi missing) - the engine will serve on CPU/Vulkan.'
                Write-Host '>>>       install the driver from https://www.nvidia.com/drivers (GeForce or Studio branch), reboot,'
                Write-Host '>>>       then: pallama engine update  (auto-picks the newest CUDA build the driver supports; runtimes bundled)'
            } elseif ($nvidia) {
                Write-Host '>>> GPU preflight: NVIDIA driver present - CUDA engine lane eligible (pallama engine update picks the newest driver-compatible CUDA build)'
            }
        }
    }

    # One-click readiness: persist config migrations (legacy api_keys ->
    # [[keys]]). Best-effort: never fail the install on it.
    try {
        & $installedExe migrate *> $null
        if ($LASTEXITCODE -eq 0) { Write-Host '>>> config migrated/verified (canonical form)' }
        else { Write-Host '>>> config migration skipped (run: pallama migrate)' }
    } catch { Write-Host '>>> config migration skipped (run: pallama migrate)' }
    # One-click readiness: bootstrap the llama.cpp engine so the box is
    # infer-ready (opt out: PALLAMA_INSTALL_ENGINE=0). Idempotent: an
    # already-active engine skips the download. Optional first model via
    # PALLAMA_INSTALL_MODEL (opt-in).
    $engineOk = $false
    try { $engineOk = -not [string]::IsNullOrEmpty((& $installedExe engine list 2>$null | Select-String '\[active\]')) } catch { }
    if ($env:PALLAMA_INSTALL_ENGINE -ne '0' -and -not $engineOk) {
        Write-Host ">>> bootstrapping llama.cpp engine (pallama engine update - largest download of this install)..."
        try {
            & $installedExe engine update --no-gate
            if ($LASTEXITCODE -eq 0) { Write-Host '>>> engine bootstrap complete' }
            else { Write-Host ">>> WARN: engine bootstrap failed - inference NOT ready. Run: pallama engine update" }
        } catch { Write-Host ">>> WARN: engine bootstrap failed - inference NOT ready. Run: pallama engine update" }
    } else {
        Write-Host '>>> engine already active (or bootstrap disabled) - skipping engine download'
    }
    if ($env:PALLAMA_INSTALL_MODEL) {
        Write-Host ">>> pulling first model: $env:PALLAMA_INSTALL_MODEL..."
        try {
            & $installedExe pull $env:PALLAMA_INSTALL_MODEL
            if ($LASTEXITCODE -eq 0) { Write-Host ">>> model ready: $env:PALLAMA_INSTALL_MODEL" }
            else { Write-Host ">>> WARN: model pull failed - run: pallama pull $env:PALLAMA_INSTALL_MODEL" }
        } catch { Write-Host ">>> WARN: model pull failed - run: pallama pull $env:PALLAMA_INSTALL_MODEL" }
    }
    if ($WithService) { Register-PallamaTask $installedExe }
    elseif ($taskExisted) {
        # F155b: binary was replaced under an existing task — bring the
        # daemon back up on the new build (Register path re-starts it).
        Start-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Write-Host ">>> restarted scheduled task '$TaskName' on the upgraded binary"
    }
    Write-Host '>>> system ready - check health: pallama doctor'
    Write-Host '>>> All inference is upstream llama.cpp, mistral.rs and SGLang - the engine authors did the hard parts.'
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
