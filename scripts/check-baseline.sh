#!/usr/bin/env bash
# Builds the two helpers for this computer and checks that their native libraries (llama.cpp, whisper.cpp) were
# configured with the fixed CPU baseline and, on a Mac, the macOS 11 target, and (release only) were optimized: both
# CMAKE_C_FLAGS_RELEASE and CMAKE_CXX_FLAGS_RELEASE hold -O2 (.cargo/config.toml; cmake-rs alone strips /O on Windows and
# the libraries are built at /Od). Exits 1 if any check failed.
#
#   scripts/check-baseline.sh [release|debug]     (default: release)
#
# The optimized check is made for release only: a debug build of the helpers uses another CMake configuration
# (RelWithDebInfo on the Mac) that does not read CMAKE_<LANG>_FLAGS_RELEASE, so there it only prints the flags.
#
# Which CMake cache to read is asked of cargo itself (the `out_dir` of each build script it ran), never guessed from
# target/*/build/*: that folder keeps the caches of older builds.
set -euo pipefail

cd "$(dirname "$0")/.."
profile="${1:-release}"
case "$profile" in
  release) cargo_flags=(--release); dir=release ;;
  debug) cargo_flags=(); dir=debug ;;
  *) echo "usage: $0 [release|debug]" >&2; exit 2 ;;
esac

failures=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; failures=$((failures + 1)); }

# check_cache <crate> <helper>: the CMake cache of <crate> (the native library built for <helper>).
check_cache() {
  local crate="$1" helper="$2" out cache flags
  out="$(cargo build ${cargo_flags[@]+"${cargo_flags[@]}"} -p "$helper" --message-format=json |
    jq -r --arg c "$crate" 'select(.reason == "build-script-executed" and (.package_id | (contains("#" + $c + "@") or startswith($c + " ")))) | .out_dir' |
    tail -n 1)"
  cache="$out/build/CMakeCache.txt"
  if [[ -z "$out" || ! -f "$cache" ]]; then
    fail "$helper: no CMake cache for $crate (looked in '${out:-nothing}/build')"
    return
  fi

  if grep -qx 'GGML_NATIVE:BOOL=OFF' "$cache"; then
    pass "$helper: $crate has GGML_NATIVE:BOOL=OFF"
  else
    fail "$helper: $crate has $(grep -m1 '^GGML_NATIVE:' "$cache" || echo 'no GGML_NATIVE entry')"
  fi

  local lang value
  for lang in C CXX; do
    value="$(sed -n "s/^CMAKE_${lang}_FLAGS_RELEASE:[A-Z]*=//p" "$cache")"
    if [[ "$profile" != release ]]; then
      printf 'INFO  %s: %s has CMAKE_%s_FLAGS_RELEASE=%s (not checked in a debug build)\n' "$helper" "$crate" "$lang" "$value"
    elif [[ "$value" == *-O2* || "$value" == *"/O2"* ]]; then
      pass "$helper: $crate has CMAKE_${lang}_FLAGS_RELEASE=$value"
    else
      fail "$helper: $crate is not optimized: CMAKE_${lang}_FLAGS_RELEASE='$value' has no -O2 (copy the CMAKE_*_FLAGS_RELEASE lines of .cargo/config.toml)"
    fi
  done

  if [[ "$(uname -s)" == Darwin ]]; then
    flags="$(grep -E '^CMAKE_(C|CXX)_FLAGS:' "$cache" | grep -o -- '-mmacosx-version-min=[0-9.]*' | sort -u | tr '\n' ' ')"
    if [[ "$flags" == "-mmacosx-version-min=11.0 " ]]; then
      pass "$helper: $crate compiled with -mmacosx-version-min=11.0"
    else
      fail "$helper: $crate compiled with '${flags:-no -mmacosx-version-min}', not 11.0"
    fi
  fi
}

check_cache llama-cpp-sys-2 voice-engine-llm
check_cache whisper-rs-sys voice-engine-stt

if [[ "$(uname -s)" == Darwin ]]; then
  for helper in voice-engine-llm voice-engine-stt; do
    minos="$(vtool -show-build "target/$dir/$helper" | awk '$1 == "minos" { print $2 }')"
    if [[ "$minos" == "11.0" ]]; then
      pass "$helper: binary says minimum macOS $minos"
    else
      fail "$helper: binary says minimum macOS '${minos:-unknown}', not 11.0"
    fi
  done
fi

if [[ "$failures" -ne 0 ]]; then
  echo "$failures check(s) failed. After changing .cargo/config.toml run: cargo clean ${cargo_flags[@]+"${cargo_flags[@]}"} -p whisper-rs-sys -p llama-cpp-sys-2" >&2
  exit 1
fi
echo "baseline OK"
