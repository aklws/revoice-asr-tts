param(
    [string]$PackageDir = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$pyprojectPath = Join-Path $repoRoot "pyproject.toml"

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

if ([string]::IsNullOrWhiteSpace($PackageDir)) {
    $appVersion = Get-AppVersionFromPyProject $pyprojectPath
    $PackageDir = Join-Path $repoRoot "dist\Revoice-ASR-TTS-v$appVersion"
}

function Test-PathOrFail {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (Test-Path $Path) {
        Write-Host "[OK] $Description -> $Path"
        return $true
    }

    Write-Host "[MISSING] $Description -> $Path" -ForegroundColor Red
    return $false
}

function Test-AstIndex {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AstIndexPath
    )

    if (-not (Test-Path $AstIndexPath)) {
        Write-Host "[MISSING] ModelScope AST 索引 -> $AstIndexPath" -ForegroundColor Red
        return $false
    }

    $raw = Get-Content -Path $AstIndexPath -Raw -Encoding UTF8
    $json = $raw | ConvertFrom-Json

    $indexCount = 0
    if ($null -ne $json.index) {
        $indexCount = $json.index.PSObject.Properties.Count
    }

    if ($indexCount -gt 0) {
        Write-Host "[OK] ModelScope AST 索引非空 -> $indexCount entries"
        return $true
    }

    Write-Host "[BAD] ModelScope AST 索引为空" -ForegroundColor Red
    return $false
}

function Test-TextPatternOrFail {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Pattern,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path $Path)) {
        Write-Host "[MISSING] $Description -> $Path" -ForegroundColor Red
        return $false
    }

    $raw = Get-Content -Path $Path -Raw -Encoding UTF8
    if ($raw -match $Pattern) {
        Write-Host "[OK] $Description"
        return $true
    }

    Write-Host "[BAD] $Description -> 未匹配: $Pattern" -ForegroundColor Red
    return $false
}

if (-not (Test-Path $PackageDir)) {
    throw "打包目录不存在: $PackageDir"
}

$checks = @()
$directBundleMode = Test-Path (Join-Path $PackageDir "funasr")

if ($directBundleMode) {
    Write-Host "检测到直接随包携带 Python 依赖，按旧模式校验情感识别相关模块。"
    $checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "funasr") -Description "funasr 包目录")
    $checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "funasr\version.txt") -Description "funasr 版本文件")
    $checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "modelscope\pipelines\audio\funasr_pipeline.py") -Description "ModelScope FunASR pipeline 模块")
    $checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "modelscope\pipelines\audio\__init__.py") -Description "ModelScope audio pipeline 包")
    $checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "modelscope\models\audio\funasr\model.py") -Description "ModelScope FunASR model 注册模块")
    $checks += @(Test-AstIndex -AstIndexPath (Join-Path $PackageDir "modelscope\utils\ast_index_file.py"))
} else {
    Write-Host "检测到运行时 uv sync 打包模式，改为校验依赖声明与运行时资源。"
    $packagePyproject = Join-Path $PackageDir "pyproject.toml"
    $checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "Launch-Revoice-ASR-TTS.bat") -Description "启动脚本")
    $checks += @(Test-PathOrFail -Path $packagePyproject -Description "打包 pyproject.toml")
    $checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "uv.lock") -Description "uv.lock")
    $checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "vendor\Fun-ASR-GGUF") -Description "Fun-ASR GGUF vendor 目录")
    $checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "vendor\MeanVC2") -Description "MeanVC2 vendor 目录")
    $checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "vendor\index-tts") -Description "IndexTTS vendor 目录")
    $checks += @(Test-TextPatternOrFail -Path $packagePyproject -Pattern '"funasr"' -Description "pyproject 依赖包含 funasr")
    $checks += @(Test-TextPatternOrFail -Path $packagePyproject -Pattern '"modelscope==1\.38\.1"' -Description "pyproject 依赖包含固定版本 modelscope")
    $checks += @(Test-TextPatternOrFail -Path $packagePyproject -Pattern '"protobuf>=6\.31\.1,<8"' -Description "pyproject 依赖包含 protobuf 约束")
}

if ($checks -contains $false) {
    Write-Host ""
    if ($directBundleMode) {
        Write-Host "验收失败：当前打包产物仍缺少情感识别所需模块。" -ForegroundColor Red
    } else {
        Write-Host "验收失败：当前运行时打包产物缺少必要依赖声明或 vendor 资源。" -ForegroundColor Red
    }
    exit 1
}

Write-Host ""
if ($directBundleMode) {
    Write-Host "验收通过：情感识别相关打包产物已就位。" -ForegroundColor Green
} else {
    Write-Host "验收通过：运行时依赖声明与语音模型资源已就位。" -ForegroundColor Green
}
