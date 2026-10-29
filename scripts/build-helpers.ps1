# Builds the engine's helper programs on Windows and names them the way Tauri expects a sidecar: <name>-<target triple>.exe.
# The reusable workflow (.github/workflows/build-helpers.yml) calls this script, so the recipe is one thing.
#
#   scripts/build-helpers.ps1 -Target x86_64-pc-windows-msvc -OutDir dist [-Helpers llm,stt] [-LlmName NAME] [-SttName NAME]
#                             [-EngineRef REF]
#
#   -Target     x86_64-pc-windows-msvc or aarch64-pc-windows-msvc. Must be the architecture of this computer: the native
#               libraries are built by CMake for the machine that runs the build.
#   -OutDir     receives <name>-<target triple>.exe for each helper, SHA256SUMS and build-info.txt.
#   -Helpers    which to build, a comma list among llm, stt, llm-vulkan and stt-vulkan (default llm,stt). The Vulkan
#               variants are x86_64-pc-windows-msvc only and need the Vulkan SDK (VULKAN_SDK set; the repository's
#               .github/actions/vulkan-sdk installs it on a runner). Their file name is <name>-vulkan-<target triple>.exe.
#   -LlmName    the sidecar's base name for the AI helper (default voice-engine-llm); -SttName the same for speech
#               (default voice-engine-stt). The engine does not fix them: the app does.
#   -EngineRef  the tag or commit being built; only written into build-info.txt.
#
# Run it from a clone of the engine: the build reads the engine's own .cargo/config.toml (the CPU baseline: GGML_NATIVE off
# and AVX2 on x64), which Cargo does not do for an app that depends on the engine as a git dependency. The script checks
# that the baseline was applied (the CMake caches) and fails if it was not.
#
# Windows ARM64 needs clang-cl and Ninja, with CC, CXX, CMAKE_GENERATOR=Ninja and CXXFLAGS=/EHsc set, in a Visual Studio ARM64
# environment: .github/actions/windows-arm64-toolchain does exactly that on a runner, and its comments say why.
param(
  [Parameter(Mandatory)][string]$Target,
  [Parameter(Mandatory)][string]$OutDir,
  [string]$Helpers = 'llm,stt',
  [string]$LlmName = 'voice-engine-llm',
  [string]$SttName = 'voice-engine-stt',
  [string]$EngineRef = '',
  # The Vulkan builds go in a short folder: the shader compiler's paths get long (the same C:\tg the CI job windows-gpu uses).
  [string]$VulkanTargetDir = 'C:\tg'
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

function Die([string]$message) { throw "build-helpers: $message" }

$hostTriple = ((rustc -vV) | Where-Object { $_ -like 'host: *' }) -replace '^host: ', ''
if ($hostTriple -ne $Target) { Die "this computer is $hostTriple, not $Target (the native libraries are built for the computer that builds them)" }
if ($Target -notin 'x86_64-pc-windows-msvc', 'aarch64-pc-windows-msvc') { Die "$Target is built by scripts/build-helpers.sh" }
$x64 = $Target -eq 'x86_64-pc-windows-msvc'
foreach ($n in $LlmName, $SttName) {
  if ($n -notmatch '^[A-Za-z0-9._-]+$') { Die "'$n' is not a usable file name (letters, digits, '.', '_' and '-')" }
}
$wanted = @($Helpers -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($wanted.Count -eq 0) { Die 'no helper asked for' }
foreach ($h in $wanted) {
  if ($h -notin 'llm', 'stt', 'llm-vulkan', 'stt-vulkan') { Die "unknown helper '$h' (llm, stt, llm-vulkan, stt-vulkan)" }
  if ($h -like '*-vulkan' -and -not $x64) { Die "$h exists for x86_64-pc-windows-msvc only" }
}
if ($x64 -and ($env:RUSTFLAGS -or $env:CARGO_ENCODED_RUSTFLAGS)) {
  Die "RUSTFLAGS is set, and it replaces the rustflags of .cargo/config.toml (the AVX2 baseline). Unset it."
}
if (-not $x64) {
  foreach ($v in 'CC', 'CXX', 'CMAKE_GENERATOR', 'CXXFLAGS') {
    if (-not (Get-Item "env:$v" -ErrorAction SilentlyContinue)) {
      Die "$v is not set. Windows ARM64 needs the toolchain of .github/actions/windows-arm64-toolchain (clang-cl, Ninja, /EHsc, the ARM64 Visual Studio environment)."
    }
  }
}
if ($wanted -like '*-vulkan' -and -not $env:VULKAN_SDK) { Die 'VULKAN_SDK is not set (the Vulkan builds need the Vulkan SDK)' }

New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $OutDir 'SHA256SUMS'), (Join-Path $OutDir 'build-info.txt')

$protocol = [regex]::Match((Get-Content -Raw 'crates/protocol/src/lib.rs'), 'PROTOCOL_VERSION: u32 = (\d+);').Groups[1].Value
if (-not $protocol) { Die 'cannot read PROTOCOL_VERSION from crates/protocol/src/lib.rs' }

# The versions Cargo.lock pins for a crate (a comma list when it pins several).
function Get-LockVersions([string]$crate) {
  $text = Get-Content -Raw 'Cargo.lock'
  ([regex]::Matches($text, "(?m)^name = `"$([regex]::Escape($crate))`"\r?\nversion = `"([^`"]+)`"") | ForEach-Object { $_.Groups[1].Value }) -join ','
}

$script:failed = $false
function Pass([string]$m) { Write-Host "PASS  $m" }
function Fail([string]$m) { Write-Host "FAIL  $m"; $script:failed = $true }

# The CMake cache of the native library this helper was built with says what the baseline is.
function Test-Baseline([string]$pkg, [string]$crate, [object[]]$messages, [bool]$vulkan) {
  $out = $messages | Where-Object { $_.reason -eq 'build-script-executed' -and $_.package_id -like "*#$crate@*" } |
    Select-Object -Last 1 -ExpandProperty out_dir
  $cache = if ($out) { Join-Path $out 'build\CMakeCache.txt' } else { '' }
  if (-not $cache -or -not (Test-Path $cache)) { Fail "${pkg}: no CMake cache for $crate (build script out_dir: '$out')"; return }
  $lines = Get-Content $cache
  # ARM64: GGML_AVX2 is an x86 option and is not in the cache there. The ARM64 baseline is ggml's own: with GGML_NATIVE off
  # and no GGML_CPU_ARM_ARCH, ggml-cpu adds no -march/-mcpu, so clang uses its default for aarch64-pc-windows-msvc
  # (armv8-a with NEON; no dot-product or i8mm kernels) whatever the build machine's CPU is.
  $checks = @(@{ Name = 'GGML_NATIVE'; Pattern = '^GGML_NATIVE:BOOL=OFF$' })
  if ($x64) { $checks += @{ Name = 'GGML_AVX2'; Pattern = '^GGML_AVX2:[A-Z]+=ON$' } }
  if ($vulkan) { $checks += @{ Name = 'GGML_VULKAN'; Pattern = '^GGML_VULKAN:[A-Z]+=ON$' } }
  foreach ($check in $checks) {
    $found = @($lines | Where-Object { $_ -match $check.Pattern })
    if ($found.Count -gt 0) { Pass "${pkg}: $crate has $($found[0])" }
    else { Fail "${pkg}: $crate needs $($check.Pattern), its cache has: $(@($lines | Where-Object { $_ -match "^$($check.Name):" }) -join ' ')" }
  }
  if (-not $x64) {
    $arch = @(Select-String -Path (Join-Path $out 'build\build.ninja') -Pattern '-m(arch|cpu)=\S+' -AllMatches -ErrorAction SilentlyContinue |
      ForEach-Object { $_.Matches.Value } | Sort-Object -Unique)
    Write-Host "INFO  ${pkg}: $crate compiled with -march/-mcpu flags: $(if ($arch) { $arch -join ' ' } else { '(none: the compiler default, armv8-a)' })"
    if ($arch -match 'native') { Fail "${pkg}: $crate compiled for the build machine's CPU" }
  }
}

# The Vulkan builds use Ninja, not the Visual Studio generator (as llama.cpp's own Windows Vulkan CI does): the
# vulkan-shaders-gen install step fails under MSBuild.
function Use-Ninja {
  if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $ninja = if (Test-Path $vswhere) { & $vswhere -latest -products * -find 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe' | Select-Object -First 1 }
    if (-not $ninja) { Die 'ninja.exe not found' }
    $env:PATH = "$(Split-Path $ninja);$env:PATH"
  }
}

$files = @()
foreach ($h in $wanted) {
  $vulkan = $h -like '*-vulkan'
  $kind = $h -replace '-vulkan$', ''
  if ($kind -eq 'llm') { $pkg = 'voice-engine-llm'; $crate = 'llama-cpp-sys-2'; $base = $LlmName }
  else { $pkg = 'voice-engine-stt'; $crate = 'whisper-rs-sys'; $base = $SttName }
  $file = if ($vulkan) { "$base-vulkan-$Target.exe" } else { "$base-$Target.exe" }
  Write-Host "==> building $pkg$(if ($vulkan) { ' (Vulkan)' }) for $Target"

  # --locked: the versions in Cargo.lock are the ones that ship. The JSON on stdout says where the binary and each CMake
  # cache are; Cargo's own progress and warnings still go to stderr.
  $cargoArgs = @('build', '--release', '--locked', '-p', $pkg, '--message-format=json-render-diagnostics')
  $savedGenerator = $env:CMAKE_GENERATOR; $savedTargetDir = $env:CARGO_TARGET_DIR
  if ($vulkan) {
    $cargoArgs += @('--features', 'vulkan')
    Use-Ninja
    $env:CMAKE_GENERATOR = 'Ninja'
    $env:CARGO_TARGET_DIR = $VulkanTargetDir
  }
  try {
    $json = & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) { Die "cargo build -p $pkg failed" }
  } finally {
    $env:CMAKE_GENERATOR = $savedGenerator; $env:CARGO_TARGET_DIR = $savedTargetDir
  }
  $messages = @($json | ForEach-Object { try { $_ | ConvertFrom-Json } catch { } })
  $exe = $messages | Where-Object {
    $_.reason -eq 'compiler-artifact' -and $_.target.name -eq $pkg -and ($_.target.kind -contains 'bin') -and $_.executable
  } | Select-Object -Last 1 -ExpandProperty executable
  if (-not $exe -or -not (Test-Path $exe)) { Die "cargo did not report a binary for $pkg" }
  Test-Baseline $pkg $crate $messages $vulkan
  Copy-Item $exe (Join-Path $OutDir $file) -Force
  $files += $file
}
if ($script:failed) { Die 'a baseline check failed; nothing was published (after changing .cargo/config.toml run: cargo clean --release -p whisper-rs-sys -p llama-cpp-sys-2)' }

# SHA256SUMS in the format sha256sum -c reads: lower case hash, two spaces, the name. LF line endings, no BOM.
$sums = ($files | ForEach-Object { "$((Get-FileHash -Algorithm SHA256 (Join-Path $OutDir $_)).Hash.ToLower())  $_" }) -join "`n"
[System.IO.File]::WriteAllText((Join-Path $OutDir 'SHA256SUMS'), "$sums`n")

$commit = (git rev-parse HEAD 2>$null)
if (-not $commit) { $commit = 'unknown' }
$info = @(
  "engine-ref: $(if ($EngineRef) { $EngineRef } else { 'not given' })",
  "engine-commit: $commit",
  "target: $Target",
  "helpers: $($files -join ' ')",
  "rustc: $(rustc -V)",
  "built: $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))",
  "protocol: $protocol",
  'native libraries (Cargo.lock):'
)
foreach ($c in 'llama-cpp-2', 'llama-cpp-sys-2', 'whisper-rs', 'whisper-rs-sys', 'parakeet-rs', 'ort', 'ort-sys') {
  $info += "  ${c}: $(Get-LockVersions $c)"
}
[System.IO.File]::WriteAllText((Join-Path $OutDir 'build-info.txt'), (($info -join "`n") + "`n"))

# The startup handshake against what was just built (the same script CI runs on the downloaded artifact).
Write-Host "==> handshake: {`"op`":`"ping`"} to each helper, expecting `"protocol`":$protocol"
$names = $wanted | ForEach-Object {
  $k = $_ -replace '-vulkan$', ''
  $b = if ($k -eq 'llm') { $LlmName } else { $SttName }
  if ($_ -like '*-vulkan') { "$b-vulkan" } else { $b }
}
& (Join-Path $PSScriptRoot 'smoke-helpers.ps1') -Dir $OutDir -Triple $Target -Names $names -ExpectedProtocol ([int]$protocol)
if ($LASTEXITCODE -ne 0) { Die 'the handshake failed' }
Write-Host "built $($files.Count) helper(s) into $OutDir"
