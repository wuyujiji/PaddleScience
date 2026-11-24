#!/usr/bin/env bash
set -euo pipefail

stage1_pass_cases="allen_cahn,climateformer,control_arm,deepcfd,deephpms,fsi,gencast,heart,hpinns,ifm,meteoformer,ntopo,preformer,smc_reac,transformer4sr"
stage2_pass_cases="bubble,darcy,earthformer,heat_pinn,ldc,phygeonet,pipe,xpinn"
stage3_pass_cases="graphcast,operator_learning,regae,unetformer,perovskite_solar_cells,xrdmatch"
stage4_pass_cases="extformer_moe"
stage5_pass_cases="gpinn,ide,laplace,quick_start"
# Collect every stageN_pass_cases group, so adding a new stage needs no change here.
all_passed_cases=""
while IFS= read -r stage_var; do
  [[ -n "${all_passed_cases}" ]] && all_passed_cases+=","
  all_passed_cases+="${!stage_var}"
done < <(compgen -A variable | grep -E '^stage[0-9]+_pass_cases$' | sort -V)
dynamic_error_cases="fpde,phylstm,tempogan"
# has the same cinn error on A100 (fourcastnet,utae)
cinn_error_cases="fourcastnet,utae,biharmonic2d,euler_beam,lorenz,phycrnet,rossler,spinn,tgcn,wgangp"
acc_fail_cases="aneurysm,bracket,cnn_uts,epnn,heat_exchanger,kmcast,nls_mb,qnm,shock_wave,topopt,velocitygan,chip_heat"
train_slower_cases=""
infer_slower_cases="cylinder,mlp_li,nsfnet,iops"
both_slower_cases="tadf"
new_cases=""

all_cases="${all_passed_cases},${dynamic_error_cases},${cinn_error_cases},${acc_fail_cases},${train_slower_cases},${infer_slower_cases},${both_slower_cases},${new_cases}"

cases_arg="stage1_pass"
warmup=5
iters=495
retry=1
device="0"

usage() {
  cat <<'EOF'
Usage: bash run_cinn_test.sh [options]

Options:
  --cases CASES    Model name, comma-separated model names, or one of:
                   stageN_pass (e.g. stage1_pass ... stage6_pass), dynamic_error,
                   all_passed, cinn_error, acc_fail, train_slower, infer_slower, both_slower, new
                   (default: stage1_pass)
  --warmup N       Warmup iterations (default: 5)
  --iters N        Timed iterations (default: 495)
  --retry N        Number of independent runs (default: 1)
  --device IDS     GPU ID or comma-separated GPU IDs (default: 0)
  -h, --help       Show this help

Examples:
  bash run_cinn_test.sh
  bash run_cinn_test.sh --cases stage2_pass
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
  stage[0-9]*_pass|stage[0-9]*_pass_cases)
    if [[ "${cases_arg}" == *_cases ]]; then
      stage_var="${cases_arg}"
    else
      stage_var="${cases_arg}_cases"
    fi
    if ! declare -p "${stage_var}" &>/dev/null; then
      echo "Error: unknown stage group '${cases_arg}'." >&2
      exit 2
    fi
    cases="${!stage_var}"
    output_name="${stage_var%_cases}"
    ;;
  all_passed|all_passed_cases)
    cases="${all_passed_cases}"
    output_name="all_passed"
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
  new|new_cases)
    cases="${new_cases}"
    output_name="new"
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

if [[ -z "${cases}" ]]; then
  echo "Error: no models configured for '${cases_arg}'." >&2
  exit 2
fi

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
