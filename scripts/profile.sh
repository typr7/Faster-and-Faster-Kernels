#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/profile.sh <relative-path-to-cu>

Example:
  scripts/profile.sh bf16_matmul/matmul_v1.cu

Environment overrides:
  CUDA_ARCH=sm_89                  CUDA architecture passed to nvcc
  PROFILE_SYMBOL=matmul_v1         Function symbol to profile; defaults to .cu basename
  EXTRA_NVCC_FLAGS="..."           Extra flags appended to nvcc
  EXTRA_NCU_ARGS="..."             Extra arguments appended to ncu
  KEEP_PROFILE_BIN=1               Keep the temporary binary directory
USAGE
}

die() {
    echo "error: $*" >&2
    exit 1
}

detect_cuda_arch() {
    if [[ -n "${CUDA_ARCH:-}" ]]; then
        echo "${CUDA_ARCH}"
        return
    fi

    local compute_cap
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:].' || true)"
    if [[ -n "${compute_cap}" ]]; then
        echo "sm_${compute_cap}"
        return
    fi

    echo "sm_89"
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

src_rel="$1"
case "${src_rel}" in
    /*) die "pass a path relative to the repo root, not an absolute path" ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

src_abs="$(realpath -m "${repo_root}/${src_rel}")"
case "${src_abs}" in
    "${repo_root}"/*) ;;
    *) die "source path must stay inside the repo: ${src_rel}" ;;
esac

[[ -f "${src_abs}" ]] || die "CUDA source not found: ${src_rel}"
[[ "${src_abs}" == *.cu ]] || die "source path must point to a .cu file: ${src_rel}"

src_dir_abs="$(dirname "${src_abs}")"
profile_entry_abs="${src_dir_abs}/profile_entry.cpp"
[[ -f "${profile_entry_abs}" ]] || die "profile entry not found next to source: ${profile_entry_abs}"

command -v nvcc >/dev/null 2>&1 || die "nvcc not found in PATH"
command -v ncu >/dev/null 2>&1 || die "ncu not found in PATH"

cd "${repo_root}"

report_dir="${repo_root}/profile"
mkdir -p "${report_dir}"

label="${src_rel%.cu}"
label="${label//\//_}"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_path="${report_dir}/${label}_${timestamp}.ncu-rep"
log_path="${report_dir}/${label}_${timestamp}.log"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/faster-kernels-profile.XXXXXX")"
cleanup() {
    if [[ "${KEEP_PROFILE_BIN:-0}" != "1" ]]; then
        rm -rf "${tmp_dir}"
    else
        echo "kept temporary binary directory: ${tmp_dir}"
    fi
}
trap cleanup EXIT

binary_path="${tmp_dir}/${label}_profile"
cuda_arch="$(detect_cuda_arch)"
profile_symbol="${PROFILE_SYMBOL:-$(basename "${src_abs}" .cu)}"

if [[ ! "${profile_symbol}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    die "PROFILE_SYMBOL is not a valid C++ identifier: ${profile_symbol}"
fi

extra_nvcc_flags=()
if [[ -n "${EXTRA_NVCC_FLAGS:-}" ]]; then
    read -r -a extra_nvcc_flags <<< "${EXTRA_NVCC_FLAGS}"
fi

extra_ncu_args=()
if [[ -n "${EXTRA_NCU_ARGS:-}" ]]; then
    read -r -a extra_ncu_args <<< "${EXTRA_NCU_ARGS}"
fi

echo "Compiling ${src_rel} with ${profile_entry_abs#${repo_root}/}"
echo "CUDA arch: ${cuda_arch}"
echo "Profile symbol: ${profile_symbol}"
nvcc \
    -std=c++17 \
    -O3 \
    -lineinfo \
    "-arch=${cuda_arch}" \
    "-DPROFILE_MATMUL_SYMBOL=${profile_symbol}" \
    "${extra_nvcc_flags[@]}" \
    "${src_abs}" \
    "${profile_entry_abs}" \
    -o "${binary_path}"

echo "Binary: ${binary_path}"
echo "Profiling with Nsight Compute full metric set"
echo "Report: ${report_path}"
echo "Log: ${log_path}"

ncu \
    --profile-from-start off \
    --set full \
    --force-overwrite \
    -o "${report_path}" \
    "${extra_ncu_args[@]}" \
    "${binary_path}" 2>&1 | tee "${log_path}"
