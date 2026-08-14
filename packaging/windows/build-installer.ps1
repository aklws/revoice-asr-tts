param(
    [switch]$RebuildApp
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$pyprojectPath = Join-Path $repoRoot "pyproject.toml"
$buildScript = Join-Path $scriptDir "build.ps1"
$innoScript = Join-Path $scriptDir "installer.iss"

$appName = "Revoice ASR-TTS"
$appId = "Revoice-ASR-TTS"
$packageLauncherName = "Launch-Revoice-ASR-TTS.bat"
$distRoot = Join-Path $repoRoot "dist"
$appIcon = Join-Path $repoRoot "ui\assets\revoice_asr_tts.ico"
$runtimeIconRelativePath = "ui\assets\revoice_asr_tts.ico"
$maxSingleInstallerBytes = 4GB

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

function Get-AppVersionFromPyProject([string]$filePath) {
    if (-not (Test-Path $filePath)) {
        throw "未找到 pyproject.toml: $filePath"
    }

    foreach ($line in Get-Content $filePath -Encoding UTF8) {
        if ($line -match '^\s*version\s*=\s*"([^"]+)"\s*$') {
            return $Matches[1]
        }
    }

    throw "无法从 pyproject.toml 解析版本号: $filePath"
}

$appVersion = Get-AppVersionFromPyProject $pyprojectPath
$packageName = "Revoice-ASR-TTS-v$appVersion"
$packageDir = Join-Path $distRoot $packageName
$packageLauncher = Join-Path $packageDir $packageLauncherName
$outputInstaller = Join-Path $distRoot "$packageName-Setup.exe"

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
    $sizeBytes = $measure.Sum
    if ($null -eq $sizeBytes) {
        return 0
    }
    return [int64]$sizeBytes
}

if (-not $innoCommand) {
    throw "未找到 ISCC.exe。请先安装 Inno Setup 6 (winget install JRSoftware.InnoSetup)，或确认它位于 PATH / Program Files / Chocolatey 常见目录中。"
}

if ($RebuildApp) {
    Write-Host "Rebuilding application package with Cython bundle packaging ..."
    & $buildScript
    if ($LASTEXITCODE -ne 0) {
        throw "Application build failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $innoScript)) {
    throw "Inno Setup 脚本不存在: $innoScript"
}

if (-not (Test-Path $packageDir)) {
    throw "应用目录不存在: $packageDir。请先运行 build.ps1，或加上 -RebuildApp。"
}

if (-not (Test-Path $packageLauncher)) {
    throw "应用启动脚本不存在: $packageLauncher"
}

$packageSizeBytes = Get-DirectorySizeBytes $packageDir
if ($packageSizeBytes -ge $maxSingleInstallerBytes) {
    $packageSizeGb = [Math]::Round($packageSizeBytes / 1GB, 2)
    Write-Warning "应用目录原始大小为 $packageSizeGb GB，超过 4GB。继续尝试 Inno Setup 压缩；是否能生成单文件安装器，以最终 Setup.exe 大小为准。"
}

Remove-FileIfExists $outputInstaller

$innoArgs = @(
    "/Qp",
    "/DAPP_NAME=$appName",
    "/DAPP_VERSION=$appVersion",
    "/DAPP_LAUNCHER=$packageLauncherName",
    "/DAPP_ID=$appId",
    "/DOUT_DIR=$distRoot",
    "/DOUT_FILENAME=$($packageName)-Setup",
    "/DPACKAGE_DIR=$packageDir",
    "/DAPP_ICON_FILE=$runtimeIconRelativePath"
)

if (Test-Path $appIcon) {
    $innoArgs += "/DINSTALLER_ICON=$appIcon"
}

$innoArgs += $innoScript

Write-Host ("Building single-file installer with Inno Setup (package size: {0:N2} GB) ..." -f ($packageSizeBytes / 1GB))
& $innoCommand.FullName @innoArgs
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup 打包失败，退出码: $LASTEXITCODE"
}

if (-not (Test-Path $outputInstaller)) {
    throw "Inno Setup 已完成，但未找到输出安装包: $outputInstaller"
}

$installerSizeBytes = (Get-Item $outputInstaller).Length
if ($installerSizeBytes -ge $maxSingleInstallerBytes) {
    $installerSizeGb = [Math]::Round($installerSizeBytes / 1GB, 2)
    throw "最终 Setup.exe 大小为 $installerSizeGb GB，已超过单文件安装器 4GB 上限。请改用分卷打包，或继续缩减内容后重试。"
}

Write-Host "Installer created: $outputInstaller"
