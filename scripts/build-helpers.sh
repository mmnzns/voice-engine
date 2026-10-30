#!/usr/bin/env bash
# Builds the engine's helper programs for this Mac and names them the way Tauri expects a sidecar: <name>-<target triple>.
# The reusable workflow (.github/workflows/build-helpers.yml) calls this script, so the recipe is one thing.
#
#   scripts/build-helpers.sh <target triple> <outdir> [--helpers llm,stt] [--llm-name NAME] [--stt-name NAME]
#                                                     [--engine-ref REF]
#
#   <target triple>  aarch64-apple-darwin. Must be the triple of this computer: the native libraries are built by CMake
#                    for the machine that runs the build. Windows is scripts/build-helpers.ps1.
#   <outdir>         receives <name>-<target triple> for each helper, SHA256SUMS and build-info.txt.
#   --helpers        which to build, a comma list of llm and stt (default llm,stt). The Vulkan variants (llm-vulkan,
#                    stt-vulkan) exist for x86_64-pc-windows-msvc only: build-helpers.ps1.
#   --llm-name       the sidecar's base name for the AI helper (default voice-engine-llm); --stt-name the same for speech
#                    (default voice-engine-stt). The engine does not fix them: the app does.
#   --engine-ref     the tag or commit being built; only written into build-info.txt.
#
# Run it from a clone of the engine: the build reads the engine's own .cargo/config.toml (the CPU baseline and the macOS 11
# target), which Cargo does not do for an app that depends on the engine as a git dependency. The script also checks that
# the baseline was applied (the CMake caches and the binaries' minimum macOS) and fails if it was not.
set -euo pipefail

cd "$(dirname "$0")/.."

die() { printf 'build-helpers: %s\n' "$1" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2; exit 2; }

[[ $# -ge 2 ]] || usage
target="$1"
outdir="$2"
shift 2
helpers="llm,stt"
llm_name="voice-engine-llm"
stt_name="voice-engine-stt"
engine_ref=""
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --helpers) helpers="$2" ;;
    --llm-name) llm_name="$2" ;;
    --stt-name) stt_name="$2" ;;
    --engine-ref) engine_ref="$2" ;;
    *) usage ;;
  esac
  shift 2
done

host="$(rustc -vV | sed -n 's/^host: //p')"
[[ "$target" == "$host" ]] || die "this computer is $host, not $target (the native libraries are built for the computer that builds them)"
case "$target" in
  aarch64-apple-darwin | x86_64-apple-darwin | *-linux-gnu) ;;
  *) die "$target is built by scripts/build-helpers.ps1" ;;
esac
for n in "$llm_name" "$stt_name"; do
  [[ "$n" =~ ^[A-Za-z0-9._-]+$ ]] || die "'$n' is not a usable file name (letters, digits, '.', '_' and '-')"
done
IFS=',' read -r -a wanted <<<"$helpers"
for h in ${wanted[@]+"${wanted[@]}"}; do
  case "$h" in
    llm | stt) ;;
    llm-vulkan | stt-vulkan) die "$h exists for x86_64-pc-windows-msvc only (scripts/build-helpers.ps1)" ;;
    *) die "unknown helper '$h' (llm, stt, llm-vulkan, stt-vulkan)" ;;
  esac
done
[[ ${#wanted[@]} -gt 0 ]] || die "no helper asked for"

mkdir -p "$outdir"
outdir="$(cd "$outdir" && pwd)"
rm -f "$outdir/SHA256SUMS" "$outdir/build-info.txt"
json="$(mktemp)"
trap 'rm -f "$json"' EXIT

expected_protocol="$(sed -n 's/.*PROTOCOL_VERSION: u32 = \([0-9]*\);.*/\1/p' crates/protocol/src/lib.rs)"
[[ -n "$expected_protocol" ]] || die "cannot read PROTOCOL_VERSION from crates/protocol/src/lib.rs"

failures=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; failures=$((failures + 1)); }

# The versions Cargo.lock pins for a crate (comma list when it pins several).
lock_versions() {
  awk -v n="$1" '
    $1 == "name" && $2 == "=" { f = ($3 == "\"" n "\""); next }
    f && $1 == "version" { gsub(/"/, "", $3); v = v (v ? "," : "") $3; f = 0 }
    END { print v }' Cargo.lock
}

# check_baseline <helper package> <native crate> <binary>: the CMake cache of the native library this helper was built
# with says GGML_NATIVE is off, it was optimized (-O2 in CMAKE_C_FLAGS_RELEASE and CMAKE_CXX_FLAGS_RELEASE), it was
# compiled for macOS 11.0, and so was the binary.
check_baseline() {
  local pkg="$1" crate="$2" bin="$3" out cache flags minos
  out="$(jq -r --arg c "$crate" 'select(.reason == "build-script-executed" and (.package_id | (contains("#" + $c + "@") or startswith($c + " ")))) | .out_dir' "$json" | tail -n 1)"
  cache="$out/build/CMakeCache.txt"
  if [[ -z "$out" || ! -f "$cache" ]]; then
    fail "$pkg: no CMake cache for $crate (looked in '${out:-nothing}/build')"
    return
  fi
  if grep -qx 'GGML_NATIVE:BOOL=OFF' "$cache"; then
    pass "$pkg: $crate has GGML_NATIVE:BOOL=OFF"
  else
    fail "$pkg: $crate has $(grep -m1 '^GGML_NATIVE:' "$cache" || echo 'no GGML_NATIVE entry')"
  fi
  # Always a release build here. cmake-rs alone strips /O under the Visual Studio generator (Windows builds at /Od); the
  # two CMAKE_*_FLAGS_RELEASE lines of .cargo/config.toml put -O2 back.
  local lang value
  for lang in C CXX; do
    value="$(sed -n "s/^CMAKE_${lang}_FLAGS_RELEASE:[A-Z]*=//p" "$cache")"
    if [[ "$value" == *-O2* || "$value" == *"/O2"* ]]; then
      pass "$pkg: $crate has CMAKE_${lang}_FLAGS_RELEASE=$value"
    else
      fail "$pkg: $crate is not optimized: CMAKE_${lang}_FLAGS_RELEASE='$value' has no -O2"
    fi
  done
  # Not checked, only shown: what the cache says for OpenMP, llamafile and the build type.
  printf 'INFO  %s: %s cache: %s\n' "$pkg" "$crate" "$(grep -E '^(GGML_OPENMP|GGML_LLAMAFILE|CMAKE_BUILD_TYPE|CMAKE_GENERATOR):' "$cache" | tr '\n' ' ')"
  if [[ "$(uname -s)" == Darwin ]]; then
    flags="$(grep -E '^CMAKE_(C|CXX)_FLAGS:' "$cache" | grep -o -- '-mmacosx-version-min=[0-9.]*' | sort -u | tr '\n' ' ')"
    if [[ "$flags" == "-mmacosx-version-min=11.0 " ]]; then
      pass "$pkg: $crate compiled with -mmacosx-version-min=11.0"
    else
      fail "$pkg: $crate compiled with '${flags:-no -mmacosx-version-min}', not 11.0"
    fi
    minos="$(vtool -show-build "$bin" | awk '$1 == "minos" { print $2 }')"
    if [[ "$minos" == "11.0" ]]; then
      pass "$pkg: the binary says minimum macOS $minos"
    else
      fail "$pkg: the binary says minimum macOS '${minos:-unknown}', not 11.0"
    fi
  fi
}

# handshake <file>: the first line a helper answers to {"op":"ping"} on stdin (and nothing else on stdin until it has
# answered, so it cannot exit first). A helper that dies at startup is started again, up to 3 times: on a Mac,
# several helpers starting at once can make one die inside Apple's Metal shader cache.
handshake() {
  local bin="$1" line="" attempt
  for attempt in 1 2 3; do
    line="$({ printf '{"op":"ping"}\n'; sleep 2; } | perl -e 'alarm 120; exec @ARGV' "$bin" 2>/dev/null | head -n 1 || true)"
    [[ -n "$line" ]] && break
  done
  printf '%s' "$line"
}

files=()
for h in "${wanted[@]}"; do
  case "$h" in
    llm) pkg=voice-engine-llm crate=llama-cpp-sys-2 name="$llm_name" ;;
    stt) pkg=voice-engine-stt crate=whisper-rs-sys name="$stt_name" ;;
  esac
  echo "==> building $pkg for $target"
  # The native library is always built again from CMake: its build script is not re-run when .cargo/config.toml changes
  # (whisper-rs-sys has no rerun-if-env-changed), so a restored or older build folder can hold a library compiled
  # without the settings the checks below ask for (-O2, the baseline).
  cargo clean --release -p "$crate"
  # --locked: the versions in Cargo.lock are the ones that ship. The JSON on stdout says where the binary and each
  # CMake cache are; Cargo's own progress and warnings still go to stderr.
  cargo build --release --locked -p "$pkg" --message-format=json-render-diagnostics >"$json"
  bin="$(jq -r --arg p "$pkg" 'select(.reason == "compiler-artifact" and .target.name == $p and (.target.kind | index("bin")) and .executable != null) | .executable' "$json" | tail -n 1)"
  [[ -n "$bin" && -f "$bin" ]] || die "cargo did not report a binary for $pkg"
  check_baseline "$pkg" "$crate" "$bin"
  cp "$bin" "$outdir/$name-$target"
  chmod +x "$outdir/$name-$target"
  files+=("$name-$target")
done
[[ "$failures" -eq 0 ]] || die "$failures baseline check(s) failed; nothing was published (after changing .cargo/config.toml run: cargo clean --release -p whisper-rs-sys -p llama-cpp-sys-2)"

if command -v sha256sum >/dev/null; then sha() { sha256sum "$@"; }; else sha() { shasum -a 256 "$@"; }; fi
(cd "$outdir" && sha "${files[@]}" >SHA256SUMS)

{
  echo "engine-ref: ${engine_ref:-not given}"
  echo "engine-commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "target: $target"
  echo "helpers: ${files[*]}"
  echo "rustc: $(rustc -V)"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "protocol: $expected_protocol"
  echo "native libraries (Cargo.lock):"
  for c in llama-cpp-2 llama-cpp-sys-2 whisper-rs whisper-rs-sys parakeet-rs ort ort-sys; do
    echo "  $c: $(lock_versions "$c")"
  done
} >"$outdir/build-info.txt"

echo "==> handshake: {\"op\":\"ping\"} to each helper, expecting \"protocol\":$expected_protocol"
for f in "${files[@]}"; do
  line="$(handshake "$outdir/$f")"
  printf '%s -> %s\n' "$f" "${line:-no answer}"
  if [[ "$line" == *'"event":"ready"'* && "$line" == *"\"protocol\":$expected_protocol"[,\}]* ]]; then
    pass "$f answers ready with protocol $expected_protocol"
  else
    fail "$f did not answer ready with protocol $expected_protocol"
  fi
done
[[ "$failures" -eq 0 ]] || die "$failures check(s) failed"
echo "built ${#files[@]} helper(s) into $outdir"
