param(
    [string]$PackageDir = "D:\work\pywork\dist\Revoice-ASR-TTS-v0.0.1"
)

$ErrorActionPreference = "Stop"

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

if (-not (Test-Path $PackageDir)) {
    throw "打包目录不存在: $PackageDir"
}

$checks = @()
$checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "funasr") -Description "funasr 包目录")
$checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "funasr\version.txt") -Description "funasr 版本文件")
$checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "modelscope\pipelines\audio\funasr_pipeline.py") -Description "ModelScope FunASR pipeline 模块")
$checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "modelscope\pipelines\audio\__init__.py") -Description "ModelScope audio pipeline 包")
$checks += @(Test-PathOrFail -Path (Join-Path $PackageDir "modelscope\models\audio\funasr\model.py") -Description "ModelScope FunASR model 注册模块")
$checks += @(Test-AstIndex -AstIndexPath (Join-Path $PackageDir "modelscope\utils\ast_index_file.py"))

if ($checks -contains $false) {
    Write-Host ""
    Write-Host "验收失败：当前打包产物仍缺少情感识别所需模块。" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "验收通过：情感识别相关打包产物已就位。" -ForegroundColor Green
