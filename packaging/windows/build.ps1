$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$entryScript = Join-Path $repoRoot "main.py"
$appVersion = "0.0.5"
$fileVersion = "0.0.5.0"
$packageName = "Revoice-ASR-TTS-v$appVersion"
$distRoot = Join-Path $repoRoot "dist"
$packageDir = Join-Path $distRoot $packageName
$buildRoot = Join-Path $repoRoot "build\pyinstaller"
$specRoot = Join-Path $buildRoot "spec"
$workRoot = Join-Path $buildRoot "work"
$versionFile = Join-Path $buildRoot "version_info.txt"
$appIcon = Join-Path $repoRoot "ui\assets\revoice_asr_tts.ico"
$appIconPng = Join-Path $repoRoot "ui\assets\revoice_asr_tts.png"
$readmeSource = Join-Path $repoRoot "docs\PACKAGING.md"
$uiAssetsDir = Join-Path $repoRoot "ui\assets"
$assetDir = Join-Path $repoRoot "asset"
$qwenRootDir = Join-Path $repoRoot "bin\Qwen3-ASR-Transcribe"
$qwenPackageDir = Join-Path $qwenRootDir "qwen_asr_gguf"
$qwenModelDir = Join-Path $qwenRootDir "model"
$qwenReadme = Join-Path $qwenRootDir "readme.md"
$qwenTranscribeExe = Join-Path $qwenRootDir "transcribe.exe"
$ffmpegDir = Join-Path $repoRoot "bin\ffmpeg"
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

function New-PyInstallerVersionFile([string]$targetFile) {
    $content = @"
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=($($fileVersion.Replace('.', ', '))),
    prodvers=($($fileVersion.Replace('.', ', '))),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '080404B0',
        [
          StringStruct('CompanyName', 'Revoice'),
          StringStruct('FileDescription', 'Revoice ASR-TTS'),
          StringStruct('FileVersion', '$fileVersion'),
          StringStruct('InternalName', '$packageName'),
          StringStruct('OriginalFilename', '$packageName.exe'),
          StringStruct('ProductName', 'Revoice ASR-TTS'),
          StringStruct('ProductVersion', '$appVersion')
        ]
      )
    ]),
    VarFileInfo([VarStruct('Translation', [2052, 1200])])
  ]
)
"@
    Set-Content -Path $targetFile -Value $content -Encoding ASCII
}

function Add-OptionalPyInstallerFileArg([ref]$argsRef, [string]$optionName, [string]$sourcePath, [string]$destinationPath) {
    if (Test-Path $sourcePath) {
        $argsRef.Value += "$optionName=$sourcePath`:$destinationPath"
    }
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
        "ffmpeg*.exe",
        "transcribe.exe"
    )
    foreach ($pattern in $blockedNamePatterns) {
        if ($fileName -like $pattern) {
            return $false
        }
    }

    $blockedPathFragments = @(
        "\PySide6\",
        "\pandas\",
        "\numpy\",
        "\pyarrow\",
        "\torch\",
        "\triton\",
        "\nvidia\",
        "\bin\ffmpeg\",
        "\bin\Qwen3-ASR-Transcribe\"
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

function Remove-RedundantPyArrowRootDlls([string]$packageRoot) {
    $pyarrowDir = Join-Path $packageRoot "pyarrow"
    if (-not (Test-Path $pyarrowDir)) {
        return
    }

    $patterns = @("arrow*.dll", "parquet*.dll")
    foreach ($pattern in $patterns) {
        $rootDlls = Get-ChildItem -Path $packageRoot -Filter $pattern -File -ErrorAction SilentlyContinue
        foreach ($rootDll in $rootDlls) {
            $packageDll = Join-Path $pyarrowDir $rootDll.Name
            if (Test-Path $packageDll) {
                Remove-Item $rootDll.FullName -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Write-Host "Building $packageName with PyInstaller ..."

if (-not $upxCommand) {
    throw "未找到 upx，可先安装或把 upx 加入 PATH。"
}

Remove-DirectorySafely $packageDir
Remove-DirectorySafely $buildRoot

New-Item -ItemType Directory -Force -Path $distRoot | Out-Null
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
New-Item -ItemType Directory -Force -Path $specRoot | Out-Null
New-Item -ItemType Directory -Force -Path $workRoot | Out-Null

if (-not (Test-Path $appIcon) -and (Test-Path $appIconPng)) {
    Write-Host "Generating Windows icon from $appIconPng ..."
    uv run --with pillow python -c "from PIL import Image; from pathlib import Path; src=Path(r'$appIconPng'); dst=Path(r'$appIcon'); dst.parent.mkdir(parents=True, exist_ok=True); Image.open(src).convert('RGBA').save(dst, format='ICO', sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate Windows icon: $appIcon"
    }
} elseif (-not (Test-Path $appIcon)) {
    Write-Warning "Windows icon file not found: $appIcon"
}

New-PyInstallerVersionFile $versionFile

$pyiArgs = @(
    "run",
    "--with", "pyinstaller>=6.15",
    "--with", "pillow",
    "python",
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--clean",
    "--noupx",
    "--onedir",
    "--contents-directory=.",
    "--distpath=$distRoot",
    "--workpath=$workRoot",
    "--specpath=$specRoot",
    "--name=$packageName",
    "--version-file=$versionFile",
    "--windowed",
    "--paths=$repoRoot",
    "--paths=$qwenRootDir",
    "--exclude-module=gradio",
    "--exclude-module=gradio_client",
    "--exclude-module=fastapi",
    "--exclude-module=starlette",
    # "--exclude-module=IPython",
    # "--exclude-module=ipykernel",
    # "--exclude-module=jedi",
    "--exclude-module=expecttest",
    "--exclude-module=notebook",
    "--exclude-module=jupyter_client",
    "--exclude-module=jupyter_core",
    # "--exclude-module=sklearn",
    # "--exclude-module=pytest",
    "--collect-all=indextts",
    "--collect-all=flash_attn",
    "--collect-all=triton",
    "--collect-all=omegaconf",
    "--collect-all=torchcodec",
    "--collect-all=audiotools",
    "--collect-all=wetext",
    "--collect-all=contractions",
    "--collect-submodules=antlr4",
    "--collect-submodules=funasr",
    "--collect-data=funasr",
    "--hidden-import=yaml",
    "--hidden-import=srt",
    "--hidden-import=ui.app",
    "--hidden-import=app.core.model_setup",
    "--hidden-import=modelscope.hub.snapshot_download",
    "--hidden-import=modelscope.pipelines.audio.funasr_pipeline",
    "--hidden-import=modelscope.models.audio.funasr.model",
    "--hidden-import=flash_attn_2_cuda",
    "--copy-metadata=g2p-en",
    "--copy-metadata=flash_attn",
    "--copy-metadata=torchcodec",
    "--copy-metadata=descript-audiotools",
    "--copy-metadata=wetext",
    "--copy-metadata=contractions",
    "--copy-metadata=transformers",
    "--add-data=$uiAssetsDir`:ui/assets",
    "--add-data=$qwenModelDir`:bin/Qwen3-ASR-Transcribe/model",
    "--add-data=$ffmpegDir`:bin/ffmpeg"
)

if (Test-Path $appIcon) {
    $pyiArgs += "--icon=$appIcon"
}

Add-OptionalPyInstallerFileArg ([ref]$pyiArgs) "--add-data" $readmeSource "."
Add-OptionalPyInstallerFileArg ([ref]$pyiArgs) "--add-data" $qwenReadme "bin/Qwen3-ASR-Transcribe"
Add-OptionalPyInstallerFileArg ([ref]$pyiArgs) "--add-data" $qwenPackageDir "bin/Qwen3-ASR-Transcribe/qwen_asr_gguf"
Add-OptionalPyInstallerFileArg ([ref]$pyiArgs) "--add-binary" $qwenTranscribeExe "bin/Qwen3-ASR-Transcribe"

$pyiArgs += $entryScript

uv @pyiArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $packageDir)) {
    throw "PyInstaller output directory not found: $packageDir"
}

$outputsDir = Join-Path $packageDir "outputs"
$checkpointsDir = Join-Path $packageDir "checkpoints"
$logsDir = Join-Path $packageDir "logs"

New-Item -ItemType Directory -Force -Path $outputsDir | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointsDir | Out-Null
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

if (-not (Test-Path (Join-Path $packageDir "asset"))) {
    New-Item -ItemType Directory -Force -Path (Join-Path $packageDir "asset") | Out-Null
}

Remove-RedundantPyArrowRootDlls $packageDir
Invoke-UpxCompression $packageDir $upxCommand.Source

Write-Host "Build completed: $packageDir"
