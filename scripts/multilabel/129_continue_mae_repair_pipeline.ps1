param([string]$NccHost = 'ncc_serve_4090', [string]$H20TransferAlias = 'huoshan_TriDim', [switch]$PrefixTransferOnly)
$ErrorActionPreference = 'Stop'
$taskH20 = '/home/wangzw/mae_repair_20261008'
$taskNcc = '/tmp/wangzw_mae_repair_20261008'
$taskLogRoot = Join-Path $PSScriptRoot '../../outputs/server_sync/mae_repair_20261008'
New-Item -ItemType Directory -Force -Path $taskLogRoot | Out-Null
Start-Transcript -Path (Join-Path $taskLogRoot 'pipeline_controller.log') -Append

function Invoke-Checked([string]$Executable, [string[]]$TaskArguments) {
    $ErrorActionPreference = 'Continue'
    for ($taskAttempt = 1; $taskAttempt -le 3; $taskAttempt++) {
        $taskOutput = & $Executable @TaskArguments 2>&1
        if ($LASTEXITCODE -eq 0) { return $taskOutput }
        if ($taskAttempt -lt 3) { Start-Sleep -Seconds 4 }
    }
    throw "$Executable failed: $($taskOutput -join ' ')"
}
function Invoke-Remote([string]$Kind, [string]$Command) {
    if ($Kind -eq 'h20') {
        return Invoke-Checked 'ssh' @('-p', '10022', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'wangzw@124.174.8.252', $Command)
    }
    return Invoke-Checked 'ssh' @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', $NccHost, $Command)
}
function Wait-Marker([string]$Kind, [string]$Root, [string]$Complete, [string]$Failed) {
    $taskDeadline = (Get-Date).AddHours(24)
    do {
        $taskState = (Invoke-Remote $Kind "if test -f $Root/$Failed; then echo failed; elif test -f $Root/$Complete; then echo ready; else echo waiting; fi") -join "`n"
        if ($taskState.Trim() -eq 'failed') { throw "Upstream failed: $Kind $Root/$Failed" }
        if ($taskState.Trim() -eq 'ready') { return }
        if ((Get-Date) -gt $taskDeadline) { throw "Timed out waiting for $Kind $Complete" }
        Start-Sleep -Seconds 30
    } while ($true)
}
function Verify-Copy([string]$Source, [string]$Target, [string[]]$Files) {
    foreach ($taskFile in $Files) {
        $taskSourceHash = ((Invoke-Remote 'ncc' "sha256sum $Source/$taskFile") -join ' ') -split '\s+'
        $taskTargetHash = ((Invoke-Remote 'h20' "sha256sum $Target/$taskFile") -join ' ') -split '\s+'
        if ($taskSourceHash[0] -notmatch '^[a-f0-9]{64}$' -or $taskSourceHash[0] -ne $taskTargetHash[0]) {
            throw "Transferred file hash mismatch: $Target/$taskFile"
        }
    }
}

try {
    # SFTP relay uses aliases only after verifying the user-provided H20 endpoint.
    $taskAlias = (Invoke-Checked 'ssh' @('-G', $H20TransferAlias)) -join "`n"
    if ($taskAlias -notmatch '(?m)^hostname 124\.174\.8\.252\r?$' -or $taskAlias -notmatch '(?m)^user wangzw\r?$' -or $taskAlias -notmatch '(?m)^port 10022\r?$') {
        throw 'H20 transfer alias must resolve to wangzw@124.174.8.252:10022'
    }
    if ($PrefixTransferOnly) {
        $null = Invoke-Remote 'h20' "test -f $taskH20/outputs/video_stage_a_v2/VIDEO_STAGE_A_TRANSFER_VERIFIED"
        Write-Host 'Resuming video-prefix transfer only; existing server queues remain untouched.'
    } else {
    Write-Host 'Waiting for six full-data label-free Stage-A runs.'
    Wait-Marker 'h20' "$taskH20/outputs/stage_a_v2" 'STAGE_A_FORMAL_COMPLETE' 'STAGE_A_QUEUE_FAILED'
    Wait-Marker 'ncc' "$taskNcc/outputs/stage_a_v2" 'STAGE_A_FORMAL_COMPLETE' 'STAGE_A_QUEUE_FAILED'
    $taskVideoRoot = "$taskH20/outputs/video_stage_a_v2"
    foreach ($taskProtocol in @('cross_day', 'within_subject_day')) {
        $taskSource = "$taskNcc/outputs/stage_a_v2/formal/$taskProtocol/video_seed_240800"
        $taskParent = "$taskVideoRoot/$taskProtocol"
        $null = Invoke-Remote 'h20' "mkdir -p $taskParent"
        $null = Invoke-Checked 'scp' @('-3', '-r', '-o', 'BatchMode=yes', "${NccHost}:$taskSource", "${H20TransferAlias}:$taskParent/")
        Verify-Copy $taskSource "$taskParent/video_seed_240800" @('checkpoint.pt', 'window_embeddings.npz', 'config.json')
        Write-Host "Verified video Stage-A copy: $taskProtocol"
    }
    $null = Invoke-Remote 'h20' "touch $taskVideoRoot/VIDEO_STAGE_A_TRANSFER_VERIFIED"
    $null = Invoke-Remote 'h20' "test -f $taskH20/outputs/R0_UNMASKED_CONTRACT_VERIFIED"
    $null = Invoke-Remote 'ncc' "test -f $taskNcc/outputs/R0_VIDEO_UNMASKED_CONTRACT_VERIFIED"
    $taskH20Launch = "nohup bash $taskH20/scripts/multilabel/130_queue_mae_repair_downstream.sh > $taskH20/outputs/logs/downstream_launch.log 2>&1 < /dev/null &"
    $taskNccLaunch = "nohup bash $taskNcc/scripts/multilabel/131_prepare_mae_repair_prefixes.sh ncc > $taskNcc/outputs/logs/prefix_launch.log 2>&1 < /dev/null &"
    $null = Invoke-Remote 'ncc' $taskNccLaunch
    $null = Invoke-Remote 'h20' $taskH20Launch
    Write-Host 'Frozen downstream queue and video-prefix preparation started.'
    }
    Wait-Marker 'ncc' "$taskNcc/outputs/prefixes_v2" 'PREFIX_NCC_COMPLETE' 'PREFIX_PREPARATION_FAILED'
    foreach ($taskProtocol in @('cross_day', 'within_subject_day')) {
        $taskSource = "$taskNcc/outputs/prefixes_v2/pretrained/$taskProtocol/video"
        $taskParent = "$taskH20/outputs/prefixes_v2/pretrained/$taskProtocol"
        $null = Invoke-Remote 'h20' "mkdir -p $taskParent"
        $null = Invoke-Checked 'scp' @('-3', '-r', '-o', 'BatchMode=yes', "${NccHost}:$taskSource", "${H20TransferAlias}:$taskParent/")
        Verify-Copy $taskSource "$taskParent/video" @('prefix.npy', 'tail.pt', 'encoder_initialization.pt', 'window_embeddings.npz', 'manifest.json')
        Write-Host "Verified pretrained video prefix: $taskProtocol"
    }
    $null = Invoke-Remote 'h20' "touch $taskH20/outputs/prefixes_v2/VIDEO_PREFIX_TRANSFER_VERIFIED"
    Write-Host 'Transfer handoff complete. H20 continues E1/W1/V1 -> M2/M3/M4 -> M5-F -> M5-FT; originals remain on ncc.'
} catch {
    $taskFailure = $_
    try { $null = Invoke-Remote 'h20' "touch $taskH20/outputs/VIDEO_PREFIX_TRANSFER_FAILED" } catch { Write-Warning 'H20 unreachable while recording transfer failure.' }
    Write-Error $taskFailure
    exit 1
} finally {
    Stop-Transcript
}
