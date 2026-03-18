#!/usr/bin/env bash
set -euo pipefail

# Neuro-symbolic STL benchmark sweep
#
# Updated benchmark matrix:
#   ex1: obs 1..6
#   ex2: obs 1..6
#   ex3: obs 4
#   ex4: obs 3
#   ex5: obs 5
#
# across the four solvers: gnn, cnn, mlp, stlpy.
#
# Assumption:
#   Run this script from the neurosymbolic/ repository root.
#   Checkpoints are expected under results/ with names:
#     results/ckpt_ex1.pt ... results/ckpt_ex5.pt
#     results/cnn_ckpt_ex1.pt ... results/cnn_ckpt_ex5.pt
#     results/mlp_ckpt_ex1.pt ... results/mlp_ckpt_ex5.pt
#
# Example usage:
#   bash benchmark/run_benchmark_sweep.sh
#
# Notes:
# - topk is fixed to 1000 by default below.
# - Output directory naming follows:
#     gnn   -> results/eval_ex{ex}_obs{obs}
#     cnn   -> results/cnn_eval_ex{ex}_obs{obs}
#     mlp   -> results/mlp_eval_ex{ex}_obs{obs}
#     stlpy -> results/stlpy_eval_ex{ex}_obs{obs}

ROOT_DIR="${ROOT_DIR:-results}"
N="${N:-100}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"
TOPK="${TOPK:-1000}"
GUROBI_OUT="${GUROBI_OUT:-0}"
SAVE_PNG="${SAVE_PNG:-0}"
RUN_MICP_BASELINE="${RUN_MICP_BASELINE:-0}"
SUCCESS_METRIC="${SUCCESS_METRIC:-QP_opt}"
H="${H:-64}"
W="${W:-64}"

mkdir -p "${ROOT_DIR}"

obstacles_for_example() {
  case "$1" in
    1|2) echo "1 2 3 4 5 6" ;;
    3) echo "4" ;;
    4) echo "3" ;;
    5) echo "5" ;;
    *) echo "Unsupported example: $1" >&2; exit 1 ;;
  esac
}

get_ckpt_path() {
  local solver="$1"
  local ex="$2"
  local path=""

  case "${solver}" in
    gnn) path="results/ckpt_ex${ex}.pt" ;;
    cnn) path="results/cnn_ckpt_ex${ex}.pt" ;;
    mlp) path="results/mlp_ckpt_ex${ex}.pt" ;;
    *) echo "Unsupported learned solver: ${solver}" >&2; exit 1 ;;
  esac

  if [[ ! -f "${path}" ]]; then
    echo "Checkpoint not found: ${path}" >&2
    exit 1
  fi
  echo "${path}"
}

run_gnn() {
  local ex="$1"
  local obs="$2"
  local out_dir="${ROOT_DIR}/eval_ex${ex}_obs${obs}"
  local ckpt
  ckpt="$(get_ckpt_path gnn "${ex}")"

  python -m gnn_train.eval_qp \
    --ckpt "${ckpt}" \
    --example_id "${ex}" \
    --n "${N}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --n_obstacles "${obs}" \
    --topk "${TOPK}" \
    --success_metric "${SUCCESS_METRIC}" \
    --run_micp_baseline "${RUN_MICP_BASELINE}" \
    --gurobi_out "${GUROBI_OUT}" \
    --save_png "${SAVE_PNG}" \
    --png_dir "${out_dir}"
}

run_cnn() {
  local ex="$1"
  local obs="$2"
  local out_dir="${ROOT_DIR}/cnn_eval_ex${ex}_obs${obs}"
  local ckpt
  ckpt="$(get_ckpt_path cnn "${ex}")"

  python -m cnn_train.cnn_eval_qp \
    --ckpt "${ckpt}" \
    --example_id "${ex}" \
    --n "${N}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --n_obstacles "${obs}" \
    --topk "${TOPK}" \
    --success_metric "${SUCCESS_METRIC}" \
    --run_micp_baseline "${RUN_MICP_BASELINE}" \
    --gurobi_out "${GUROBI_OUT}" \
    --save_png "${SAVE_PNG}" \
    --png_dir "${out_dir}" \
    --H "${H}" \
    --W "${W}"
}

run_mlp() {
  local ex="$1"
  local obs="$2"
  local out_dir="${ROOT_DIR}/mlp_eval_ex${ex}_obs${obs}"
  local ckpt
  ckpt="$(get_ckpt_path mlp "${ex}")"

  python -m mlp_train.mlp_eval_qp \
    --ckpt "${ckpt}" \
    --example_id "${ex}" \
    --n "${N}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --n_obstacles "${obs}" \
    --topk "${TOPK}" \
    --success_metric "${SUCCESS_METRIC}" \
    --run_micp_baseline "${RUN_MICP_BASELINE}" \
    --gurobi_out "${GUROBI_OUT}" \
    --save_png "${SAVE_PNG}" \
    --png_dir "${out_dir}" \
    --H "${H}" \
    --W "${W}"
}

run_stlpy() {
  local ex="$1"
  local obs="$2"
  local out_dir="${ROOT_DIR}/stlpy_eval_ex${ex}_obs${obs}"

  python -m stlpy_eval.eval_micp \
    --example_id "${ex}" \
    --n "${N}" \
    --seed "${SEED}" \
    --n_obstacles "${obs}" \
    --gurobi_out "${GUROBI_OUT}" \
    --save_png "${SAVE_PNG}" \
    --png_dir "${out_dir}"
}

total_runs() {
  local total=0
  local ex
  local obs
  for ex in 1 2 3 4 5; do
    for obs in $(obstacles_for_example "${ex}"); do
      total=$((total + 4))
    done
  done
  echo "${total}"
}

RUN_TOTAL="$(total_runs)"
RUN_IDX=0

for solver in gnn cnn mlp stlpy; do
  for ex in 1 2 3 4 5; do
    for obs in $(obstacles_for_example "${ex}"); do
      RUN_IDX=$((RUN_IDX + 1))
      echo "============================================================"
      echo "[${RUN_IDX}/${RUN_TOTAL}] solver=${solver}  example=${ex}  obstacles=${obs}"
      echo "============================================================"

      case "${solver}" in
        gnn) run_gnn "${ex}" "${obs}" ;;
        cnn) run_cnn "${ex}" "${obs}" ;;
        mlp) run_mlp "${ex}" "${obs}" ;;
        stlpy) run_stlpy "${ex}" "${obs}" ;;
        *) echo "Unsupported solver: ${solver}" >&2; exit 1 ;;
      esac
    done
  done
done

echo
echo "Sweep complete. Next steps:"
echo "  python -m benchmark.result_parser --root_dir ${ROOT_DIR}"
echo "  python -m benchmark.table_results --root_dir ${ROOT_DIR}"
echo "  python -m benchmark.plot_results --root_dir ${ROOT_DIR}"

python -m benchmark.result_parser --root_dir results
python -m benchmark.table_results --root_dir results
python -m benchmark.plot_results --root_dir results