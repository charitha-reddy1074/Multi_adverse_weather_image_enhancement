param([string[]]$Datasets = @(), [int]$TimeoutSeconds = 900)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot '.venv/Scripts/python.exe'
$rawRoot = Join-Path $repoRoot 'data/raw'
$logRoot = Join-Path $repoRoot 'outputs/downloads'
New-Item -ItemType Directory -Force -Path $rawRoot,$logRoot | Out-Null
$rows = Import-Csv -LiteralPath (Join-Path $repoRoot 'datasets/downloads.csv')
$results = @()
foreach ($row in $rows) {
    if ($row.dataset -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid dataset identifier' }
    if ($Datasets.Count -gt 0 -and $row.dataset -notin $Datasets) { continue }
    if ($row.url -notlike 'https://*') { throw 'Only HTTPS download URLs are accepted' }
    if ([IO.Path]::GetFileName($row.filename) -ne $row.filename) { throw 'Invalid archive filename' }
    $dest = Join-Path $rawRoot $row.filename
    $log = Join-Path $logRoot ($row.dataset + '.log')
    $status = 'DOWNLOAD BLOCKED'
    if ($row.method -eq 'manual') {
        Write-Output "Manual source: $($row.dataset) $($row.url)"
    } else {
        if ($row.method -eq 'gdown') {
            & $pythonExe -m gdown --no-cookies --continue --timeout 30 -q $row.url -O $dest *> $log
        } elseif ($row.method -eq 'curl') {
            & curl.exe -L --fail --proto '=https' --proto-redir '=https' --max-time $TimeoutSeconds -C - $row.url -o $dest *> $log
        } else { throw "Unknown download method: $($row.method)" }
        if ($LASTEXITCODE -eq 0) { $status = 'DOWNLOADED - INTEGRITY CHECK REQUIRED' }
    }
    $hash = ''
    if ((Test-Path -LiteralPath $dest) -and $status -like 'DOWNLOADED*') {
        $hash = (Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash.ToLower()
    }
    $results += [pscustomobject]@{ dataset=$row.dataset; status=$status; url=$row.url; archive=$dest; sha256=$hash; log=$log }
    $results | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $logRoot 'status.json') -Encoding utf8
    Write-Output "$($row.dataset): $status"
}
Write-Output 'Run prepare_available_data.py for supported releases; do not use unvalidated archives for training.'
