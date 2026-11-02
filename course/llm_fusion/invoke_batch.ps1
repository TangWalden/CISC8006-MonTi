param(
    [Parameter(Mandatory=$true)][string]$RequestFile,
    [Parameter(Mandatory=$true)][string]$OutputFile,
    [Parameter(Mandatory=$true)][string]$KeyFile,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ProtocolSha256,
    [ValidateRange(1,120)][int]$MaxRequests=120,
    [ValidateRange(1,200000)][int]$MaxPromptTokens=200000,
    [ValidateRange(1,64000)][int]$MaxCompletionTokens=64000,
    [ValidateRange(0.000001,0.50)][double]$MaxUsd=0.50
)
# Local Windows transport only. Never write/decrypt credentials to a file,
# argument, environment variable, output record or exception message.
$ErrorActionPreference = 'Stop'
$taskPlainBytes=$null; $taskKey=$null; $taskHeaders=$null
$taskStream=$null; $taskWriter=$null; $taskFatal=$false
$taskUtf8 = New-Object Text.UTF8Encoding($false)
$taskPricingUrl='https://api-docs.deepseek.com/quick_start/pricing'
$taskInputPrice=0.30/1000000; $taskOutputPrice=1.20/1000000
try {
    Add-Type -AssemblyName System.Security
    $taskRequests=@()
    $taskSeen=New-Object 'System.Collections.Generic.HashSet[string]'
    foreach ($taskLine in [IO.File]::ReadAllLines($RequestFile,$taskUtf8)) {
        if ([String]::IsNullOrWhiteSpace($taskLine)) { throw 'BlankRequest' }
        $taskItem=$taskLine | ConvertFrom-Json
        if ((@($taskItem.PSObject.Properties.Name | Sort-Object) -join ',') -ne 'max_tokens,messages,request_id') { throw 'RequestSchema' }
        if ([string]$taskItem.request_id -notmatch '^[a-z0-9_-]{1,100}$' -or -not $taskSeen.Add([string]$taskItem.request_id)) { throw 'RequestIdentity' }
        if ($taskItem.max_tokens -notin @(64,512)) { throw 'RequestTokens' }
        if (@($taskItem.messages).Count -ne 2) { throw 'RequestMessages' }
        for ($taskIndex=0; $taskIndex -lt 2; $taskIndex++) {
            $taskMessage=$taskItem.messages[$taskIndex]
            if ((@($taskMessage.PSObject.Properties.Name | Sort-Object) -join ',') -ne 'content,role') { throw 'MessageSchema' }
            if ($taskMessage.role -ne @('system','user')[$taskIndex] -or $taskMessage.content -isnot [string] -or [String]::IsNullOrWhiteSpace($taskMessage.content)) { throw 'MessageContent' }
        }
        $taskRequests += $taskItem
    }
    if ($taskRequests.Count -gt 64) { throw 'BatchTooLarge' }
    # Holding this handle prevents a second writer. Existing records are only
    # appended; Python creates a new private run directory for every execution.
    $taskStream=[IO.File]::Open($OutputFile,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::Read)
    $taskReader=[IO.StreamReader]::new($taskStream,$taskUtf8,$true,1024,$true)
    $taskOld=$taskReader.ReadToEnd(); $taskReader.Dispose()
    $taskCalls=0; $taskPrompt=0L; $taskCompletion=0L; $taskStop=$false
    $taskPreviousIds=New-Object 'System.Collections.Generic.HashSet[string]'
    foreach ($taskLine in ($taskOld -split "`n")) {
        if ([String]::IsNullOrWhiteSpace($taskLine)) { continue }
        $taskOldRecord=$taskLine | ConvertFrom-Json
        if ($taskOldRecord.protocol_sha256 -ne $ProtocolSha256) { throw 'LedgerProtocolMismatch' }
        [void]$taskPreviousIds.Add([string]$taskOldRecord.request_id)
        if ($taskOldRecord.attempted -eq $true) {
            $taskCalls++; $taskPrompt += [long]$taskOldRecord.charged_prompt_tokens
            $taskCompletion += [long]$taskOldRecord.charged_completion_tokens
        }
        if ($taskOldRecord.http_status -in @(401,403) -or $taskOldRecord.reservation_exceeded -eq $true) { $taskStop=$true }
    }
    foreach ($taskItem in $taskRequests) {
        if ($taskPreviousIds.Contains([string]$taskItem.request_id)) { throw 'RepeatedLogicalRequest' }
    }
    [void]$taskStream.Seek(0,[IO.SeekOrigin]::End)
    $taskWriter=[IO.StreamWriter]::new($taskStream,$taskUtf8,1024,$true)
    $taskWriter.AutoFlush=$true
    [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
    foreach ($taskItem in $taskRequests) {
        $taskBodyObject=[ordered]@{model='deepseek-flash';thinking=@{type='disabled'};temperature=0;stream=$false;response_format=@{type='json_object'};max_tokens=[int]$taskItem.max_tokens;messages=$taskItem.messages}
        $taskBody=$taskUtf8.GetBytes(($taskBodyObject | ConvertTo-Json -Depth 20 -Compress))
        # A conservative token reservation: serialized UTF-8 bytes, plus chat
        # framing allowance. Failed/unknown-usage attempts keep the reservation.
        $taskReservePrompt=[long]$taskBody.Length+1024
        $taskReserveCompletion=[long]$taskItem.max_tokens
        for ($taskAttempt=1; $taskAttempt -le 2; $taskAttempt++) {
            $taskProjectedUsd=($taskPrompt+$taskReservePrompt)*$taskInputPrice+($taskCompletion+$taskReserveCompletion)*$taskOutputPrice
            if ($taskStop -or $taskCalls -ge $MaxRequests -or $taskPrompt+$taskReservePrompt -gt $MaxPromptTokens -or $taskCompletion+$taskReserveCompletion -gt $MaxCompletionTokens -or $taskProjectedUsd -gt $MaxUsd) {
                $taskWriter.WriteLine(([ordered]@{request_id=$taskItem.request_id;protocol_sha256=$ProtocolSha256;attempted=$false;status='blocked_budget_or_prior_fatal';utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress))
                break
            }
            if ($null -eq $taskHeaders) {
                $taskEntropy=$taskUtf8.GetBytes('MonTi-CISC8006-DeepSeek-v1')
                $taskPlainBytes=[Security.Cryptography.ProtectedData]::Unprotect([IO.File]::ReadAllBytes($KeyFile),$taskEntropy,[Security.Cryptography.DataProtectionScope]::CurrentUser)
                $taskKey=$taskUtf8.GetString($taskPlainBytes)
                $taskHeaders=@{Authorization=('Bearer '+$taskKey)}
            }
            $taskStarted=[DateTime]::UtcNow; $taskHttp=$null; $taskRetry=$false
            $taskRecord=[ordered]@{request_id=$taskItem.request_id;attempt=$taskAttempt;attempted=$true;protocol_sha256=$ProtocolSha256;started_utc=$taskStarted.ToString('o');request=$taskBodyObject;reserved_prompt_tokens=$taskReservePrompt;reserved_completion_tokens=$taskReserveCompletion;charged_prompt_tokens=$taskReservePrompt;charged_completion_tokens=$taskReserveCompletion;usage_verified=$false;pricing_url=$taskPricingUrl;input_cache_miss_usd_per_million=0.30;output_usd_per_million=1.20}
            $taskCalls++
            try {
                $taskResponse=Invoke-WebRequest -UseBasicParsing -Method Post -Uri 'https://api.deepseek.com/chat/completions' -Headers $taskHeaders -ContentType 'application/json; charset=utf-8' -Body $taskBody -TimeoutSec 60
                $taskHttp=[int]$taskResponse.StatusCode
                $taskData=$taskResponse.Content | ConvertFrom-Json
                $taskRecord.status='completed'; $taskRecord.http_status=$taskHttp
                $taskRecord.response=$taskData
                $taskRecord.model=$taskData.model; $taskRecord.response_id=$taskData.id
                $taskRecord.finish_reason=$taskData.choices[0].finish_reason
                $taskRecord.usage=$taskData.usage
                $taskP=0L; $taskC=0L
                if ([long]::TryParse([string]$taskData.usage.prompt_tokens,[ref]$taskP) -and [long]::TryParse([string]$taskData.usage.completion_tokens,[ref]$taskC) -and $taskP -ge 0 -and $taskC -ge 0) {
                    $taskRecord.charged_prompt_tokens=$taskP; $taskRecord.charged_completion_tokens=$taskC; $taskRecord.usage_verified=$true
                    if ($taskP -gt $taskReservePrompt -or $taskC -gt $taskReserveCompletion) { $taskRecord.reservation_exceeded=$true; $taskStop=$true }
                }
            } catch {
                try { $taskHttp=[int]$_.Exception.Response.StatusCode } catch {}
                $taskRecord.status='failed'; $taskRecord.http_status=$taskHttp
                $taskRecord.error_type=$_.Exception.GetType().Name
                # Never serialize exception messages, HTTP error bodies or headers.
                $taskRetry=($taskHttp -eq 429 -or ($taskHttp -ge 500 -and $taskHttp -le 599))
                if ($taskHttp -eq 401 -or $taskHttp -eq 403) { $taskStop=$true }
            }
            $taskPrompt += [long]$taskRecord.charged_prompt_tokens
            $taskCompletion += [long]$taskRecord.charged_completion_tokens
            $taskRecord.completed_utc=[DateTime]::UtcNow.ToString('o')
            $taskRecord.charged_usd_upper_bound=$taskRecord.charged_prompt_tokens*$taskInputPrice+$taskRecord.charged_completion_tokens*$taskOutputPrice
            $taskWriter.WriteLine(($taskRecord | ConvertTo-Json -Depth 30 -Compress))
            $taskStream.Flush()
            if (-not $taskRetry -or $taskAttempt -eq 2) { break }
            Start-Sleep -Seconds 2
        }
    }
    [ordered]@{status='batch_recorded';attempts_total=$taskCalls;charged_prompt_tokens=$taskPrompt;charged_completion_tokens=$taskCompletion;charged_usd_upper_bound=$taskPrompt*$taskInputPrice+$taskCompletion*$taskOutputPrice} | ConvertTo-Json -Compress
} catch {
    # Safe status only: plaintext credentials or provider error bodies must not
    # reach console logs even when the transport cannot initialize.
    $taskFatal=$true
    [ordered]@{status='transport_failed';error_type=$_.Exception.GetType().Name;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress
} finally {
    if ($taskPlainBytes) { [Array]::Clear($taskPlainBytes,0,$taskPlainBytes.Length) }
    if ($taskHeaders) { $taskHeaders.Clear() }
    $taskKey=$null; $taskPlainBytes=$null; $taskHeaders=$null
    if ($taskWriter) { $taskWriter.Dispose() }
    if ($taskStream) { $taskStream.Dispose() }
}
if ($taskFatal) { exit 1 }
