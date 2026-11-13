#!/usr/bin/env bash
set -euo pipefail

pass_cases="allen_cahn,climateformer,control_arm,deepcfd,deephpms,fsi,gencast,heart,hpinns,ifm,meteoformer,ntopo,preformer,smc_reac,transformer4sr"
dynamic_error_cases="earthformer,extformer_moe,graphcast,qnm"
cinn_error_cases="pipe"
acc_fail_cases="bracket,cnn_uts,darcy,epnn,heat_exchanger,kmcast,perovskite_solar_cells,shock_wave,topopt,xrdmatch"
train_slower_cases="bubble,mlp_li,operator_learning"
infer_slower_cases="aneurysm,cylinder,heat_pinn,ldc,nls_mb,nsfnet,phygeonet,regae,xpinn"
both_slower_cases="iops"

all_cases="${pass_cases},${dynamic_error_cases},${cinn_error_cases},${acc_fail_cases},${train_slower_cases},${infer_slower_cases},${both_slower_cases}"

cases_arg="pass"
warmup=5
iters=495
retry=1
device="0"

usage() {
  cat <<'EOF'
Usage: bash run_cinn_test.sh [options]

Options:
  --cases CASES    Model name, comma-separated model names, or one of:
                   pass, dynamic_error, cinn_error, acc_fail,
                   train_slower, infer_slower, both_slower
                   (default: pass)
  --warmup N       Warmup iterations (default: 5)
  --iters N        Timed iterations (default: 495)
  --retry N        Number of independent runs (default: 1)
  --device IDS     GPU ID or comma-separated GPU IDs (default: 0)
  -h, --help       Show this help

Examples:
  bash run_cinn_test.sh
  bash run_cinn_test.sh --cases allen_cahn
  bash run_cinn_test.sh --cases allen_cahn,climateformer --retry 2
  bash run_cinn_test.sh --cases acc_fail --warmup 5 --iters 495 --device 0,1,2,3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases)
      cases_arg="${2:?missing value for --cases}"
      shift 2
      ;;
    --cases=*)
      cases_arg="${1#*=}"
      shift
      ;;
    --warmup)
      warmup="${2:?missing value for --warmup}"
      shift 2
      ;;
    --warmup=*)
      warmup="${1#*=}"
      shift
      ;;
    --iters)
      iters="${2:?missing value for --iters}"
      shift 2
      ;;
    --iters=*)
      iters="${1#*=}"
      shift
      ;;
    --retry)
      retry="${2:?missing value for --retry}"
      shift 2
      ;;
    --retry=*)
      retry="${1#*=}"
      shift
      ;;
    --device)
      device="${2:?missing value for --device}"
      shift 2
      ;;
    --device=*)
      device="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "${warmup}" =~ ^[0-9]+$ ]]; then
  echo "Error: --warmup must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "${iters}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --iters must be a positive integer." >&2
  exit 2
fi
if [[ ! "${retry}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --retry must be a positive integer." >&2
  exit 2
fi
if [[ ! "${device}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Error: --device must be a GPU ID or comma-separated GPU IDs." >&2
  exit 2
fi

cases_arg="${cases_arg//，/,}"
cases_arg="${cases_arg// /}"
case "${cases_arg}" in
  pass|pass_cases)
    cases="${pass_cases}"
    output_name="pass"
    ;;
  dynamic_error|dynamic_error_cases)
    cases="${dynamic_error_cases}"
    output_name="dynamic_error"
    ;;
  cinn_error|cinn_error_cases)
    cases="${cinn_error_cases}"
    output_name="cinn_error"
    ;;
  acc_fail|acc_fail_cases)
    cases="${acc_fail_cases}"
    output_name="acc_fail"
    ;;
  train_slower|train_slower_cases)
    cases="${train_slower_cases}"
    output_name="train_slower"
    ;;
  infer_slower|infer_slower_cases)
    cases="${infer_slower_cases}"
    output_name="infer_slower"
    ;;
  both_slower|both_slower_cases)
    cases="${both_slower_cases}"
    output_name="both_slower"
    ;;
  *)
    cases="${cases_arg}"
    output_name="${cases_arg//,/_}"
    IFS=',' read -ra requested_cases <<< "${cases}"
    for case_name in "${requested_cases[@]}"; do
      if [[ ! ",${all_cases}," == *",${case_name},"* ]]; then
        echo "Error: unsupported model '${case_name}'." >&2
        exit 2
      fi
    done
    ;;
esac

device_name="${device//,/_}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

for ((run = 1; run <= retry; run++)); do
  output_dir="output/${output_name}_warmup${warmup}_iters${iters}_retry${run}_gpu${device_name}"
  rm -rf "${output_dir}"
  echo "[$run/$retry] cases=${cases} devices=${device} output=${output_dir}"
  CUDA_VISIBLE_DEVICES="${device}" \
    python3 test/cinn/run_cinn_test.py \
      --devices "${device}" \
      --cases "${cases}" \
      --stages infer,train \
      --warmup "${warmup}" \
      --iters "${iters}" \
      --output-dir "${output_dir}"
done
