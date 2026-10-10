param(
    [string]$NccHost = 'ncc_serve_4090',
    [string]$H20Host = 'huoshan_TriDim'
)
$ErrorActionPreference = 'Stop'
$sourceRoot = '/tmp/wangzw_mae_remaining_20261007/prefixes'
$targetRoot = '/home/wangzw/outputs/mae_remaining_20261007/prefixes'
$outRoot = '/home/wangzw/outputs/mae_remaining_20261007'
$logRoot = Join-Path $PSScriptRoot '../../outputs/server_sync/mae_remaining_20261007'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
Start-Transcript -Path (Join-Path $logRoot 'video_prefix_transfer.log') -Append
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit $LASTEXITCODE" }
}
try {
    foreach ($leaf in @('pretrained/cross_day', 'pretrained/within_subject_day', 'random/shared')) {
        $source = "$sourceRoot/$leaf/video"
        $parent = "$targetRoot/$leaf"
        $deadline = (Get-Date).AddHours(8)
        do {
            $state = & ssh -o BatchMode=yes -o ConnectTimeout=15 $NccHost "test -f $source/PREFIX_COMPLETE && echo ready; test ! -f $sourceRoot/PREFIX_PREPARATION_FAILED"
            if ($LASTEXITCODE -ne 0) { throw "ncc prefix failed or host unreachable: $leaf" }
            if ($state -contains 'ready') { break }
            if ((Get-Date) -gt $deadline) { throw "Timed out waiting for prefix: $leaf" }
            Start-Sleep -Seconds 20
        } while ($true)
        Invoke-Checked 'ssh' @('-o', 'BatchMode=yes', $H20Host, "mkdir -p $parent")
        Write-Host "Copying video prefix $leaf at $(Get-Date -Format o)"
        Invoke-Checked 'scp' @('-3', '-r', "${NccHost}:$source", "${H20Host}:$parent/")
        $manifestText = & ssh -o BatchMode=yes $NccHost "cat $source/manifest.json"
        if ($LASTEXITCODE -ne 0) { throw 'Cannot read source manifest' }
        $manifest = ($manifestText -join "`n") | ConvertFrom-Json
        $targetHash = & ssh -o BatchMode=yes $H20Host "sha256sum $parent/video/prefix.npy"
        if ($LASTEXITCODE -ne 0 -or ($targetHash -split '\s+')[0] -ne $manifest.prefix_sha256) {
            throw "Video prefix hash mismatch: $leaf"
        }
        Write-Host "Verified video prefix $leaf SHA256=$($manifest.prefix_sha256)"
    }
    Invoke-Checked 'ssh' @('-o', 'BatchMode=yes', $H20Host, "touch $targetRoot/VIDEO_PREFIX_TRANSFER_VERIFIED")
    Write-Host 'All three video prefixes copied and verified; H20 adaptation queue can proceed.'
} catch {
    & ssh -o BatchMode=yes -o ConnectTimeout=15 $H20Host "touch $outRoot/VIDEO_PREFIX_TRANSFER_FAILED"
    Write-Error $_
    exit 1
} finally {
    Stop-Transcript
}
