<#
.SYNOPSIS
  Downloads the model files listed in backend\models.manifest into backend\models\ with curl.exe.
  Existing files are kept; interrupted downloads resume from their .part file.

  Before downloading, the missing files' sizes are checked against free disk space, keeping
  HYPNOS_MODELS_MIN_FREE_GB (default 2) free; nothing is fetched when they do not fit.
  HYPNOS_MODELS_DIR overrides the target folder (the configs expect backend\models\).
  HF_TOKEN, when set, is sent to Hugging Face (higher rate limits). Gated repositories (FLUX's
  originals, gated = true in models.catalog.toml) need it, from an account that accepted the
  model's license on its Hugging Face page.
  A manifest repo that is an https:// URL is a plain download base (no token is sent there).

  An id that is not in the manifest is looked up in the model catalog (models.catalog.toml)
  through the hypnos binary (`hypnos --catalog-files <id>[:<quant>]`; beside this folder in a
  release archive, target\release in a checkout); `hypnos --catalog-block <id>[:<quant>]` then
  prints the config block to add. Hugging Face downloads are checked against their SHA-256 (the
  X-Linked-ETag of LFS files).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File backend\scripts\fetch-models.ps1
  powershell -ExecutionPolicy Bypass -File backend\scripts\fetch-models.ps1 moss-image
  powershell -ExecutionPolicy Bypass -File backend\scripts\fetch-models.ps1 lfm3-vl:2b:Q4_K_M
#>
param([string]$Model = "all")
$ErrorActionPreference = "Stop"
# Relative to this script, so the same paths work in the repository (backend\scripts) and in a
# release archive (<archive>\scripts, with models.manifest and models\ beside it).
$base = Resolve-Path (Join-Path $PSScriptRoot "..")
$manifest = Join-Path $base "models.manifest"
$modelsDir = if ($env:HYPNOS_MODELS_DIR) { $env:HYPNOS_MODELS_DIR } else { Join-Path $base "models" }
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

# The manifest's column layout is versioned; a newer layout needs newer scripts.
$mv = (Get-Content $manifest | Select-String -Pattern '^# manifest-version:\s*(\d+)' | Select-Object -First 1)
if ($mv -and [int]$mv.Matches[0].Groups[1].Value -gt 1) {
    Write-Error "$manifest is manifest-version $($mv.Matches[0].Groups[1].Value); this script reads version 1. Update Hypnos."
    exit 1
}
$entries = Get-Content $manifest | Where-Object { $_ -notmatch '^\s*(#|$)' } | ForEach-Object {
    $c = $_.Trim() -split '\s+'
    [pscustomobject]@{ Model = $c[0]; Repo = $c[1]; Remote = $c[2]; Local = $c[3]; Platform = $c[4] }
}
$ids = $entries.Model | Sort-Object -Unique
$catalogSpec = $null
$hypnosBin = $null
if ($Model -ne "all" -and $ids -notcontains $Model) {
    # Not in the lineup: a catalog entry, through the server binary that reads the catalog.
    foreach ($b in @("hypnos.exe", "target\release\hypnos.exe", "target\debug\hypnos.exe", "hypnos", "target/release/hypnos")) {
        $p = Join-Path $base $b
        if (Test-Path -LiteralPath $p -PathType Leaf) { $hypnosBin = $p; break }
    }
    if (-not $hypnosBin) {
        $cmd = Get-Command hypnos -ErrorAction SilentlyContinue
        if ($cmd) { $hypnosBin = $cmd.Source }
    }
    $lines = $null
    if ($hypnosBin) {
        Push-Location $base
        try { $lines = & $hypnosBin --catalog-files $Model } finally { Pop-Location }
    }
    if (-not $hypnosBin -or $LASTEXITCODE -ne 0 -or -not $lines) {
        Write-Host "[models] unknown model '$Model'. Lineup: all $($ids -join ' ')"
        if (-not $hypnosBin) { Write-Host "[models] (build the server to download catalog entries: run.bat build backend)" }
        exit 1
    }
    $entries = $lines | Where-Object { $_ -match '\S' } | ForEach-Object {
        $c = $_.Trim() -split '\s+'
        [pscustomobject]@{ Model = $c[0]; Repo = $c[1]; Remote = $c[2]; Local = $c[3]; Platform = $c[4] }
    }
    $catalogSpec = $Model
    $Model = @($entries)[0].Model
}

$script:fetched = 0; $script:kept = 0; $script:failed = 0
$queue = New-Object System.Collections.Generic.List[object]

# A Hugging Face repository, or a direct https:// download base.
function Get-FileUrl([string]$repoName, [string]$remote) {
    if ($repoName.StartsWith("https://")) { return "$repoName/$remote" }
    return "$hf/$repoName/resolve/main/$remote"
}

# The HF token header, only for Hugging Face.
function Get-Auth([string]$repoName) {
    if ($repoName.StartsWith("https://")) { return @() }
    return $auth
}

# Remember a file that is not downloaded yet.
function Add-ModelFile([string]$repoName, [string]$remote, [string]$dest) {
    if (Test-Path -LiteralPath $dest -PathType Leaf) { $script:kept++; return }
    $queue.Add([pscustomobject]@{ Repo = $repoName; Remote = $remote; Dest = $dest })
}

# Byte size from a HEAD request (the last Content-Length after Hugging Face's redirect); $null if unknown.
function Get-RemoteSize([string]$repoName, [string]$remote) {
    # No `2>$null`: with $ErrorActionPreference = "Stop", Windows PowerShell 5.1 turns redirected
    # native stderr into a terminating error. -s keeps curl quiet unless the request fails.
    $hdr = Get-Auth $repoName
    $headers = & $curl -fsSIL @hdr (Get-FileUrl $repoName $remote)
    if ($LASTEXITCODE -ne 0) { return $null }
    $size = $null
    foreach ($line in $headers) {
        if ($line -match '^content-length:\s*(\d+)') { $size = [int64]$Matches[1] }
    }
    return $size
}

# The SHA-256 Hugging Face publishes for an LFS file (its X-Linked-ETag); $null for other hosts
# and plain git files.
function Get-ExpectedSha256([string]$repoName, [string]$remote) {
    if ($repoName.StartsWith("https://")) { return $null }
    $hdr = Get-Auth $repoName
    $headers = & $curl -fsSI @hdr (Get-FileUrl $repoName $remote)
    if ($LASTEXITCODE -ne 0) { return $null }
    foreach ($line in $headers) {
        if ($line -match '^x-linked-etag:\s*"?([0-9a-f]{64})"?') { return $Matches[1] }
    }
    return $null
}

function Get-ModelFile([string]$repoName, [string]$remote, [string]$dest) {
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    Write-Host "[models] $repoName/$remote -> $dest"
    $hdr = Get-Auth $repoName
    $wantSha = Get-ExpectedSha256 $repoName $remote
    & $curl -fL --retry 3 -C - --progress-bar @hdr -o "$dest.part" (Get-FileUrl $repoName $remote)
    if ($LASTEXITCODE -eq 0 -and $wantSha) {
        $haveSha = (Get-FileHash -Algorithm SHA256 -LiteralPath "$dest.part").Hash.ToLowerInvariant()
        if ($haveSha -ne $wantSha) {
            Write-Host "[models] FAILED: $repoName/$remote does not match its SHA-256 (removed; rerun to fetch it again)"
            Remove-Item -Force -LiteralPath "$dest.part"
            $script:failed++
            return
        }
    }
    if ($LASTEXITCODE -eq 0) {
        Move-Item -Force -LiteralPath "$dest.part" -Destination $dest
        $script:fetched++
    } else {
        $code = $LASTEXITCODE
        Write-Host "[models] FAILED: $repoName/$remote (partial data kept in $dest.part; rerun to resume)"
        # curl -f exits 22 on an HTTP error: a gated repository answers 401 / 403.
        if ($code -eq 22 -and -not $repoName.StartsWith('https://')) {
            Write-Host "[models]   gated? accept the license at $hf/$repoName and set HF_TOKEN (a read token of that account)"
        }
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
            Add-ModelFile $e.Repo $item.path (Join-Path $modelsDir $rel)
        }
    } else {
        Add-ModelFile $e.Repo $e.Remote (Join-Path $modelsDir $e.Local)
    }
}

# Disk space: everything still missing (minus resumable .part data) has to fit, plus a margin.
if ($queue.Count -gt 0) {
    Write-Host "[models] checking the size of $($queue.Count) missing file(s)..."
    [int64]$need = 0; $unknown = 0
    foreach ($q in $queue) {
        $size = Get-RemoteSize $q.Repo $q.Remote
        if ($null -eq $size) { $unknown++; continue }
        $part = "$($q.Dest).part"
        $have = if (Test-Path -LiteralPath $part) { (Get-Item -LiteralPath $part).Length } else { 0 }
        $need += $size - $have
    }
    New-Item -ItemType Directory -Force $modelsDir | Out-Null
    $root = [System.IO.Path]::GetPathRoot((Resolve-Path $modelsDir).ProviderPath)
    [int64]$free = (New-Object System.IO.DriveInfo $root).AvailableFreeSpace
    $minFreeGb = if ($env:HYPNOS_MODELS_MIN_FREE_GB) { [int64]$env:HYPNOS_MODELS_MIN_FREE_GB } else { 2 }
    [int64]$margin = $minFreeGb * 1000000000
    $gb = { param($b) "{0:N1} GB" -f ($b / 1e9) }
    $extra = if ($unknown -gt 0) { " (+$unknown of unknown size)" } else { "" }
    Write-Host "[models] to download: $(& $gb $need)$extra, free: $(& $gb $free)"
    if ($need + $margin -gt $free) {
        Write-Host "[models] not enough disk space: need $(& $gb $need) plus $(& $gb $margin) kept free (HYPNOS_MODELS_MIN_FREE_GB)."
        Write-Host "         Free some space, pick one model (run.bat models <id>) or set HYPNOS_MODELS_DIR."
        exit 1
    }
}

foreach ($q in $queue) { Get-ModelFile $q.Repo $q.Remote $q.Dest }

Write-Host "[models] $($script:fetched) downloaded, $($script:kept) already present, $($script:failed) failed ($platform, $modelsDir)"
if ($catalogSpec -and $script:failed -eq 0) {
    Write-Host "[models] add it to the config (paths are relative to models_dir):"
    Write-Host "         & '$hypnosBin' --catalog-block $catalogSpec >> hypnos.toml"
}
if ($script:failed -gt 0) { exit 1 }
