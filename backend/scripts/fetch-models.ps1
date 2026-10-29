<#
.SYNOPSIS
  Downloads the model files listed in backend\models.manifest into backend\models\ with curl.exe.
  Existing files are kept; interrupted downloads resume from their .part file.

  HYPNOS_MODELS_DIR overrides the target folder (the configs expect backend\models\).
  HF_TOKEN, when set, is sent to Hugging Face (higher rate limits); the repositories are public.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File backend\scripts\fetch-models.ps1
  powershell -ExecutionPolicy Bypass -File backend\scripts\fetch-models.ps1 moss-image
#>
param([string]$Model = "all")
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$manifest = Join-Path $repo "backend\models.manifest"
$modelsDir = if ($env:HYPNOS_MODELS_DIR) { $env:HYPNOS_MODELS_DIR } else { Join-Path $repo "backend\models" }
$hf = "https://huggingface.co"

# Windows PowerShell aliases `curl` to Invoke-WebRequest (no resume); call the real binary
# (shipped with Windows 10 1803+). It also avoids the old TLS defaults of PowerShell 5.1.
$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) { $curl = Get-Command curl -CommandType Application -ErrorAction SilentlyContinue }
if (-not $curl) { Write-Host "[models] curl.exe is required (Windows 10 1803 or newer)"; exit 1 }
$curl = $curl.Source

$platform = "other"
if ($IsMacOS -and [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "Arm64") { $platform = "macos-arm64" }

$auth = @()
if ($env:HF_TOKEN) { $auth = @("-H", "Authorization: Bearer $($env:HF_TOKEN)") }

$entries = Get-Content $manifest | Where-Object { $_ -notmatch '^\s*(#|$)' } | ForEach-Object {
    $c = $_.Trim() -split '\s+'
    [pscustomobject]@{ Model = $c[0]; Repo = $c[1]; Remote = $c[2]; Local = $c[3]; Platform = $c[4] }
}
$ids = $entries.Model | Sort-Object -Unique
if ($Model -ne "all" -and $ids -notcontains $Model) {
    Write-Host "[models] unknown model '$Model'. Known: all $($ids -join ' ')"
    exit 1
}

$script:fetched = 0; $script:kept = 0; $script:failed = 0

function Get-ModelFile([string]$repoName, [string]$remote, [string]$dest) {
    if (Test-Path -LiteralPath $dest -PathType Leaf) { $script:kept++; return }
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    Write-Host "[models] $repoName/$remote -> $dest"
    & $curl -fL --retry 3 -C - --progress-bar @auth -o "$dest.part" "$hf/$repoName/resolve/main/$remote"
    if ($LASTEXITCODE -eq 0) {
        Move-Item -Force -LiteralPath "$dest.part" -Destination $dest
        $script:fetched++
    } else {
        Write-Host "[models] FAILED: $repoName/$remote (partial data kept in $dest.part; rerun to resume)"
        $script:failed++
    }
}

foreach ($e in $entries) {
    if ($Model -ne "all" -and $e.Model -ne $Model) { continue }
    if ($e.Platform -ne "all" -and $e.Platform -ne $platform) { continue }
    if ($e.Remote.EndsWith("/")) {
        # Folder: list it through the Hub API and fetch every file in it.
        $json = & $curl -fsSL @auth "$hf/api/models/$($e.Repo)/tree/main/$($e.Remote.TrimEnd('/'))"
        if ($LASTEXITCODE -ne 0) { Write-Host "[models] FAILED: could not list $($e.Repo)/$($e.Remote)"; $script:failed++; continue }
        foreach ($item in (($json -join "") | ConvertFrom-Json)) {
            if ($item.type -ne "file") { continue }
            $rel = $e.Local + $item.path.Substring($e.Remote.Length)
            Get-ModelFile $e.Repo $item.path (Join-Path $modelsDir $rel)
        }
    } else {
        Get-ModelFile $e.Repo $e.Remote (Join-Path $modelsDir $e.Local)
    }
}

Write-Host "[models] $($script:fetched) downloaded, $($script:kept) already present, $($script:failed) failed ($platform, $modelsDir)"
if ($script:failed -gt 0) { exit 1 }
