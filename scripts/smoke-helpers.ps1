# Checks built helpers the way an app starts them: the checksums in SHA256SUMS match, and each helper answers the startup
# handshake, {"op":"ping"} on stdin, with a line that says "event":"ready" and the engine's protocol version, then exits when
# its stdin closes. Works on Windows PowerShell, PowerShell 7 and PowerShell on a Mac or Linux.
#
#   scripts/smoke-helpers.ps1 -Dir <folder with the files> -Triple <target triple> -Names voice-engine-llm,voice-engine-stt
#
# -Names are the sidecar base names; the file looked for is <name>-<triple>[.exe] (a Vulkan variant is <name>-vulkan, so
# 'voice-engine-llm-vulkan' gives voice-engine-llm-vulkan-<triple>.exe). Exits 1 if anything is missing or wrong.
param(
  [Parameter(Mandatory)][string]$Dir,
  [Parameter(Mandatory)][string]$Triple,
  [Parameter(Mandatory)][string[]]$Names,
  [int]$ExpectedProtocol = 2,
  [int]$TimeoutSecs = 60
)
$ErrorActionPreference = 'Stop'
# Accepts -Names a,b (an array) and -Names 'a,b' (one string).
$Names = @($Names | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$onWindows = $env:OS -eq 'Windows_NT'
$ext = if ($onWindows) { '.exe' } else { '' }
$Dir = (Resolve-Path $Dir).Path
$failed = $false
function Fail([string]$message) { Write-Host "FAIL  $message"; $script:failed = $true }

# The checksums. The artifact lists exactly the helpers, so each wanted file must be in it.
$sums = @{}
$sumsFile = Join-Path $Dir 'SHA256SUMS'
if (Test-Path $sumsFile) {
  foreach ($l in Get-Content $sumsFile) {
    if ($l -match '^([0-9a-f]{64}) [ *](.+)$') { $sums[$Matches[2]] = $Matches[1] }
  }
} else {
  Fail "SHA256SUMS is not in $Dir"
}

foreach ($name in $Names) {
  $file = "$name-$Triple$ext"
  $path = Join-Path $Dir $file
  if (-not (Test-Path $path)) {
    Fail "$file is not in $Dir (it has: $((Get-ChildItem $Dir | ForEach-Object Name) -join ', '))"
    continue
  }
  $want = $sums[$file]
  $have = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
  if ($want -and $want -eq $have) { Write-Host "PASS  ${file}: sha256 matches SHA256SUMS" }
  else { Fail "${file}: sha256 is $have, SHA256SUMS says '$want'"; continue }

  # An artifact download does not keep the executable bit.
  if (-not $onWindows) { & chmod +x $path }

  # The handshake. A helper that dies before answering is started again, up to 3 times: on a Mac several helpers
  # starting at once can make one die inside Apple's Metal shader cache.
  $line = $null; $exit = $null; $stderr = ''
  for ($attempt = 1; $attempt -le 3 -and -not $line; $attempt++) {
    $psi = [System.Diagnostics.ProcessStartInfo]::new($path)
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    $errTask = $p.StandardError.ReadToEndAsync()
    # UTF-8 without a BOM, written as bytes: ProcessStartInfo.StandardInputEncoding does not exist on .NET Framework
    # (Windows PowerShell 5.1), and its StandardInput would use the console's code page.
    $ping = [System.Text.UTF8Encoding]::new($false).GetBytes("{`"op`":`"ping`"}`n")
    $p.StandardInput.BaseStream.Write($ping, 0, $ping.Length)
    $p.StandardInput.BaseStream.Flush()
    $lineTask = $p.StandardOutput.ReadLineAsync()
    if ($lineTask.Wait($TimeoutSecs * 1000)) { $line = $lineTask.Result } else { $p.Kill(); Fail "${file}: no answer within ${TimeoutSecs}s" }
    # The app has gone: stdin closes and the helper must exit by itself.
    try { $p.StandardInput.Close() } catch { }
    if (-not $p.WaitForExit(20000)) { $p.Kill(); if ($line) { Fail "${file}: still running 20 s after its stdin closed" } }
    else { $exit = $p.ExitCode }
    if ($errTask.Wait(5000)) { $stderr = $errTask.Result }
    if (-not $line -and $attempt -lt 3) { Write-Host "INFO  ${file}: no answer (attempt $attempt), starting it again" }
  }
  Write-Host "$file -> $line"
  $ready = $null
  try { $ready = $line | ConvertFrom-Json } catch { }
  if ($ready -and $ready.event -eq 'ready' -and $ready.protocol -eq $ExpectedProtocol) {
    Write-Host "PASS  $file answers ready with protocol $ExpectedProtocol"
  } else {
    Fail "$file did not answer ready with protocol ${ExpectedProtocol}: '$line' (stderr: $stderr)"
  }
  if ($null -ne $exit -and $exit -ne 0) { Fail "${file}: exited with $exit after its stdin closed" }
}
if ($failed) { exit 1 }
Write-Host 'smoke test OK'
