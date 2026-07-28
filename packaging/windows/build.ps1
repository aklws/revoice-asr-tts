param(
    [string]$AppVersion = "0.0.6",
    [switch]$NoUpx,
    [switch]$IncludeBundledFfmpeg,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$venvDir = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$helperScript = Join-Path $scriptDir "build_cython_bundle.py"
$entryScript = Join-Path $repoRoot "main.py"
$packageName = "Revoice-ASR-TTS-v$appVersion"
$launcherName = "Launch-Revoice-ASR-TTS.bat"
$distRoot = Join-Path $repoRoot "dist"
$packageDir = Join-Path $distRoot $packageName
$buildRoot = Join-Path $repoRoot "build\cython_bundle"
$compiledRoot = Join-Path $buildRoot "compiled"
$compiledBuildRoot = Join-Path $buildRoot "cython_build"
$appIcon = Join-Path $repoRoot "ui\assets\revoice_asr_tts.ico"
$appIconPng = Join-Path $repoRoot "ui\assets\revoice_asr_tts.png"
$projectReadmeZh = Join-Path $repoRoot "README.md"
$projectReadmeEn = Join-Path $repoRoot "README.en.md"
$pyprojectSource = Join-Path $repoRoot "pyproject.toml"
$uvLockSource = Join-Path $repoRoot "uv.lock"
$envExampleSource = Join-Path $repoRoot ".env.example"
$assetDir = Join-Path $repoRoot "asset"
$assetExcludeFiles = @("*.mp3", "*.wav", "*.ogg", "*.flac", "*.m4a", "*.aac", "*.wma")
$ffmpegDir = Join-Path $repoRoot "bin\ffmpeg"
$flashAttnDir = Join-Path $repoRoot "bin\flash_attn"
$vendorRoot = Join-Path $repoRoot "vendor"
$upxCommand = Get-Command upx -ErrorAction SilentlyContinue

function Stop-ProcessesInDirectory([string]$targetDir) {
    if (-not (Test-Path $targetDir)) {
        return
    }

    $normalizedTarget = [System.IO.Path]::GetFullPath($targetDir).TrimEnd('\') + '\'
    $lockedProcesses = Get-Process | Where-Object {
        try {
            $processPath = $_.Path
            if (-not $processPath) {
                return $false
            }
            return [System.IO.Path]::GetFullPath($processPath).StartsWith($normalizedTarget, [System.StringComparison]::OrdinalIgnoreCase)
        } catch {
            return $false
        }
    }

    foreach ($process in $lockedProcesses) {
        Write-Host "Stopping running packaged app: $($process.Name) (PID=$($process.Id))"
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }

    if ($lockedProcesses) {
        Start-Sleep -Milliseconds 500
    }
}

function Remove-DirectorySafely([string]$targetDir) {
    if (-not (Test-Path $targetDir)) {
        return
    }

    Stop-ProcessesInDirectory $targetDir

    try {
        Remove-Item $targetDir -Recurse -Force
    } catch {
        throw "无法删除目录 '$targetDir'。请确认其中的 exe、dll 或日志文件没有被占用，然后重试。原始错误: $($_.Exception.Message)"
    }
}

function Invoke-RobocopyCopy(
    [string]$sourceDir,
    [string]$targetDir,
    [string[]]$excludeDirs = @(),
    [string[]]$excludeFiles = @()
) {
    if (-not (Test-Path $sourceDir)) {
        throw "复制源目录不存在: $sourceDir"
    }

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    $robocopyArgs = @(
        $sourceDir,
        $targetDir,
        "/E",
        "/R:1",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP"
    )

    if ($excludeDirs.Count -gt 0) {
        $robocopyArgs += "/XD"
        $robocopyArgs += $excludeDirs
    }
    if ($excludeFiles.Count -gt 0) {
        $robocopyArgs += "/XF"
        $robocopyArgs += $excludeFiles
    }

    & robocopy @robocopyArgs | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -ge 8) {
        throw "robocopy 复制失败，退出码: $exitCode (source=$sourceDir target=$targetDir)"
    }
}

function Copy-FileIfExists([string]$sourcePath, [string]$targetPath) {
    if (-not (Test-Path $sourcePath)) {
        return
    }

    $targetParent = Split-Path -Parent $targetPath
    if ($targetParent) {
        New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
    }
    Copy-Item -Path $sourcePath -Destination $targetPath -Force
}

function Write-LauncherScript([string]$launcherPath) {
    $content = @"
@echo off
setlocal
set "APP_ROOT=%~dp0"
if "%APP_ROOT:~-1%"=="\" set "APP_ROOT=%APP_ROOT:~0,-1%"
where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo 未检测到 uv。请先安装 uv: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)
pushd "%APP_ROOT%"
set "UV_PROJECT_ENVIRONMENT=%APP_ROOT%\.venv"
set "PYTHONPATH=%APP_ROOT%"
set "PATH=%APP_ROOT%\bin\ffmpeg\bin;%APP_ROOT%\bin\ffmpeg;%PATH%"
set "PYTHONUTF8=1"
uv sync --frozen
if errorlevel 1 (
    echo.
    echo uv sync failed.
    popd
    exit /b 1
)
uv run python "%APP_ROOT%\main.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Revoice exited with code %EXIT_CODE%.
)
exit /b %EXIT_CODE%
"@
    Set-Content -Path $launcherPath -Value $content -Encoding Default
}

function Test-UpxSafeTarget([string]$filePath) {
    $fileName = [System.IO.Path]::GetFileName($filePath)
    $normalizedPath = $filePath.Replace('/', '\')
    $extension = [System.IO.Path]::GetExtension($fileName).ToLowerInvariant()

    if ($extension -notin @(".exe", ".dll", ".pyd")) {
        return $false
    }

    $blockedNamePatterns = @(
        "python*.dll",
        "vcruntime*.dll",
        "msvcp*.dll",
        "ucrtbase.dll",
        "api-ms-win-*.dll",
        "torch*.dll",
        "torch_*.pyd",
        "c10*.dll",
        "fbgemm*.dll",
        "libiomp*.dll",
        "cud*.dll",
        "cu*.dll",
        "nv*.dll",
        "Qt6*.dll",
        "PySide6*.pyd",
        "shiboken6*.pyd",
        "numpy*.pyd",
        "pandas*.pyd",
        "pyarrow*.pyd",
        "arrow*.dll",
        "parquet*.dll",
        "opencv*.dll",
        "ffmpeg*.exe"
    )
    foreach ($pattern in $blockedNamePatterns) {
        if ($fileName -like $pattern) {
            return $false
        }
    }

    $blockedPathFragments = @(
        "\bin\ffmpeg\"
    )
    foreach ($fragment in $blockedPathFragments) {
        if ($normalizedPath.IndexOf($fragment, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $false
        }
    }

    return $true
}

function Invoke-UpxCompression([string]$targetDir, [string]$upxExePath) {
    if (-not (Test-Path $targetDir)) {
        throw "UPX 压缩目标目录不存在: $targetDir"
    }

    $allFiles = Get-ChildItem -Path $targetDir -Recurse -File
    $candidates = @($allFiles | Where-Object { Test-UpxSafeTarget $_.FullName })
    if (-not $candidates) {
        Write-Host "UPX post-pack skipped: no safe candidates found."
        return
    }

    Write-Host "Running UPX --best --lzma on $($candidates.Count) files ..."
    $compressedCount = 0
    $skippedCount = 0

    foreach ($candidate in $candidates) {
        $upxOutput = & $upxExePath --best --lzma $candidate.FullName 2>&1
        if ($LASTEXITCODE -eq 0) {
            $compressedCount += 1
        } elseif (($upxOutput | Out-String) -match "AlreadyPackedException") {
            $skippedCount += 1
        } else {
            $skippedCount += 1
            Write-Warning "UPX skipped or failed for $($candidate.FullName)"
        }
    }

    Write-Host "UPX post-pack completed. compressed=$compressedCount skipped=$skippedCount"
}

Write-Host "Building $packageName with Cython bundle packaging ..."

if (-not (Test-Path $venvPython)) {
    throw "未找到虚拟环境 Python: $venvPython"
}
if (-not (Test-Path $helperScript)) {
    throw "未找到 Cython 打包 helper: $helperScript"
}
if (-not (Test-Path $entryScript)) {
    throw "入口脚本不存在: $entryScript"
}

Remove-DirectorySafely $packageDir
Remove-DirectorySafely $buildRoot

New-Item -ItemType Directory -Force -Path $distRoot | Out-Null
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null

if (-not (Test-Path $appIcon) -and (Test-Path $appIconPng)) {
    Write-Host "Generating Windows icon from $appIconPng ..."
    & $venvPython -c "from PIL import Image; from pathlib import Path; src=Path(r'$appIconPng'); dst=Path(r'$appIcon'); dst.parent.mkdir(parents=True, exist_ok=True); Image.open(src).convert('RGBA').save(dst, format='ICO', sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate Windows icon: $appIcon"
    }
}

$helperArgs = @(
    $helperScript,
    "--repo-root", $repoRoot,
    "--output-root", $compiledRoot,
    "--build-root", $compiledBuildRoot
)
if ($DryRun) {
    $helperArgs += "--dry-run"
}

Write-Host "Compiling local packages into pyd modules ..."
& $venvPython @helperArgs
if ($LASTEXITCODE -ne 0) {
    throw "Cython bundle build failed with exit code $LASTEXITCODE"
}

if ($DryRun) {
    Write-Host "Dry run enabled. Skipping runtime copy and installer payload assembly."
    return
}

New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

Write-Host "Copying compiled application payload ..."
Invoke-RobocopyCopy $compiledRoot $packageDir @("__pycache__", "cy", "tmp", "build")

Copy-FileIfExists $entryScript (Join-Path $packageDir "main.py")
Copy-FileIfExists $projectReadmeZh (Join-Path $packageDir "README.md")
Copy-FileIfExists $projectReadmeEn (Join-Path $packageDir "README.en.md")
Copy-FileIfExists $pyprojectSource (Join-Path $packageDir "pyproject.toml")
Copy-FileIfExists $uvLockSource (Join-Path $packageDir "uv.lock")
Copy-FileIfExists $envExampleSource (Join-Path $packageDir ".env.example")

if ($IncludeBundledFfmpeg -and (Test-Path $ffmpegDir)) {
    Invoke-RobocopyCopy $ffmpegDir (Join-Path $packageDir "bin\ffmpeg") @("__pycache__")
} elseif ($IncludeBundledFfmpeg) {
    Write-Warning "已指定 -IncludeBundledFfmpeg，但未找到本地 ffmpeg 目录: $ffmpegDir"
} else {
    Write-Host "Skipping bundled ffmpeg copy by default. Install ffmpeg separately or place it under bin\\ffmpeg manually if needed."
}
if (Test-Path $flashAttnDir) {
    Invoke-RobocopyCopy $flashAttnDir (Join-Path $packageDir "bin\flash_attn") @("__pycache__")
}
if (Test-Path $assetDir) {
    Invoke-RobocopyCopy $assetDir (Join-Path $packageDir "asset") @("__pycache__") $assetExcludeFiles
} else {
    New-Item -ItemType Directory -Force -Path (Join-Path $packageDir "asset") | Out-Null
}
if (Test-Path $vendorRoot) {
    Invoke-RobocopyCopy $vendorRoot (Join-Path $packageDir "vendor") @("__pycache__", "build", ".git", ".ipynb_checkpoints")
}

$outputsDir = Join-Path $packageDir "outputs"
$checkpointsDir = Join-Path $packageDir "checkpoints"
$logsDir = Join-Path $packageDir "logs"
$launcherPath = Join-Path $packageDir $launcherName

New-Item -ItemType Directory -Force -Path $outputsDir | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointsDir | Out-Null
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
Write-LauncherScript $launcherPath

if (-not (Test-Path $launcherPath)) {
    throw "启动脚本生成失败: $launcherPath"
}
if (-not (Test-Path (Join-Path $packageDir "pyproject.toml"))) {
    throw "pyproject.toml 复制失败: $(Join-Path $packageDir 'pyproject.toml')"
}

if ($NoUpx) {
    Write-Host "Skipping UPX post-pack because -NoUpx was specified."
} elseif ($upxCommand) {
    Invoke-UpxCompression $packageDir $upxCommand.Source
} else {
    Write-Warning "未找到 upx，跳过 post-pack 压缩。"
}

Write-Host "Build completed: $packageDir"

