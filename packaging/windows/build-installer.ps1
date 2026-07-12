param(
    [switch]$RebuildApp
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$buildScript = Join-Path $scriptDir "build.ps1"
$innoScript = Join-Path $scriptDir "installer.iss"

$appName = "Revoice ASR-TTS"
$appId = "Revoice-ASR-TTS"
$appVersion = "0.0.5"
$packageName = "Revoice-ASR-TTS-v$appVersion"
$packageExeName = "$packageName.exe"
$distRoot = Join-Path $repoRoot "dist"
$packageDir = Join-Path $distRoot $packageName
$packageExe = Join-Path $packageDir $packageExeName
$appIcon = Join-Path $repoRoot "ui\assets\revoice_asr_tts.ico"

$installerBuildRoot = Join-Path $repoRoot "build\nsis"
$payloadRoot = Join-Path $installerBuildRoot "payload"
$payloadArchive = Join-Path $payloadRoot "$packageName.7z"
$outputInstaller = Join-Path $distRoot "$packageName-Setup.exe"

function Find-FirstCommand([string[]]$names) {
    foreach ($name in $names) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command
        }
    }
    return $null
}

function Find-FirstExistingPath([string[]]$paths) {
    foreach ($path in $paths) {
        if ($path -and (Test-Path $path)) {
            return $path
        }
    }
    return $null
}

function Find-InnoSetupCommand() {
    $command = Get-Command iscc -ErrorAction SilentlyContinue
    if ($command) {
        return $command
    }

    $candidatePaths = @()
    if ($env:ProgramFiles) {
        $candidatePaths += (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    }
    if (${env:ProgramFiles(x86)}) {
        $candidatePaths += (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
    }
    if ($env:LOCALAPPDATA) {
        $candidatePaths += (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    }
    if ($env:ChocolateyInstall) {
        $candidatePaths += (Join-Path $env:ChocolateyInstall "bin\iscc.exe")
    }

    $candidatePath = Find-FirstExistingPath $candidatePaths
    if (-not $candidatePath) {
        return $null
    }

    return Get-Item $candidatePath
}

$innoCommand = Find-InnoSetupCommand

function Remove-FileIfExists([string]$path) {
    if (Test-Path $path) {
        Remove-Item $path -Force
    }
}

function Get-DirectorySizeBytes([string]$path) {
    if (-not (Test-Path $path)) {
        return 0
    }
    $measure = Get-ChildItem -Path $path -Recurse -File | Measure-Object -Property Length -Sum
    return [int64]($measure.Sum ?? 0)
}

if (-not $innoCommand) {
    throw "未找到 ISCC.exe。请先安装 Inno Setup 6 (winget install JRSoftware.InnoSetup)，或确认它位于 PATH / Program Files / Chocolatey 常见目录中。"
}

if ($RebuildApp) {
    Write-Host "Rebuilding application package with PyInstaller ..."
    & $buildScript
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $innoScript)) {
    throw "Inno Setup 脚本不存在: $innoScript"
}

if (-not (Test-Path $packageDir)) {
    throw "应用目录不存在: $packageDir。请先运行 build.ps1，或加上 -RebuildApp。"
}

if (-not (Test-Path $packageExe)) {
    throw "应用主程序不存在: $packageExe"
}

Remove-FileIfExists $outputInstaller

$innoArgs = @(
    "/Qp",
    "/DAPP_NAME=$appName",
    "/DAPP_VERSION=$appVersion",
    "/DAPP_EXE=$packageExeName",
    "/DAPP_ID=$appId",
    "/DOUT_DIR=$distRoot",
    "/DOUT_FILENAME=$($packageName)-Setup",
    "/DPACKAGE_DIR=$packageDir"
)

if (Test-Path $appIcon) {
    $innoArgs += "/DINSTALLER_ICON=$appIcon"
}

$innoArgs += $innoScript

Write-Host "Building single-file installer with Inno Setup (Ultra LZMA2 compression, this may take a while) ..."
& $innoCommand.FullName @innoArgs
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup 打包失败，退出码: $LASTEXITCODE"
}

Write-Host "Installer created: $outputInstaller"
