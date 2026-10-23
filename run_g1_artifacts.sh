#!/usr/bin/env bash
# Rebuild the G1 artifacts needed by FinalRetriever, without submitting anything.
# Usage: bash run_g1_artifacts.sh [new-output-directory]
# Run with the project's Python environment activated on a compute/GPU node.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
G1_PYTHON="${G1_PYTHON:-python}"
G1_OUTPUT="${1:-cache/g1-rerun}"

if [[ $# -gt 1 ]]; then
    echo "Usage: bash run_g1_artifacts.sh [new-output-directory]" >&2
    exit 2
fi
if [[ -e "$G1_OUTPUT" ]]; then
    echo "Output path already exists: $G1_OUTPUT" >&2
    echo "Choose a new output directory to preserve earlier results." >&2
    exit 1
fi

# Check dependencies before starting a long run or creating output files.
"$G1_PYTHON" -c 'import torch, sentence_transformers, datasets, bm25s; print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (embedding inference will be slower)")'

mkdir -p -- "$G1_OUTPUT"
G1_OUTPUT="$(cd -- "$G1_OUTPUT" && pwd)"
exec > >(tee "$G1_OUTPUT/pipeline.log") 2>&1

echo "G1 artifacts: $G1_OUTPUT"
echo "Split: test; top-20 candidate depth: 50; reservoir depth: 100; dense weight: 0.25; RRF: 60"

echo "[1/10] Extracting development ground truth"
"$G1_PYTHON" -u -m retrieval.evaluation.make_ground_truth \
    --split test --output "$G1_OUTPUT/ground_truth.json"

echo "[2/10] Goal-aware hybrid retrieval (GPU query encoding)"
"$G1_PYTHON" -u -m retrieval.run_goal_aware_hybrid_chunked \
    --split test --chunk_size 100 --topk 20 --candidate_k 50 \
    --dense_weight 0.25 --rrf_k 60 --catalog_size 47071 \
    --ground_truth "$G1_OUTPUT/ground_truth.json" \
    --output_dir "$G1_OUTPUT/goal_aware_chunks" \
    --final_predictions "$G1_OUTPUT/predictions_goal_aware_hybrid_full.json" \
    --results "$G1_OUTPUT/results_goal_aware_hybrid_full.json"

echo "[3/10] Adaptive hybrid retrieval (GPU query encoding)"
"$G1_PYTHON" -u -m retrieval.run_adaptive_goal_hybrid \
    --split test --topk 20 --candidate_k 50 --dense_weight 0.25 --rrf_k 60 \
    --output "$G1_OUTPUT/predictions_adaptive_goal_full.json"

echo "[4/10] Tuning fusion of adaptive and goal-aware rankings"
"$G1_PYTHON" -u -m retrieval.tune_prediction_fusion \
    --adaptive_predictions "$G1_OUTPUT/predictions_adaptive_goal_full.json" \
    --goal_predictions "$G1_OUTPUT/predictions_goal_aware_hybrid_full.json" \
    --ground_truth "$G1_OUTPUT/ground_truth.json" \
    --split test --catalog_size 47071 --topk 20 --rrf_k 60 \
    --output "$G1_OUTPUT/results_prediction_fusion.json" \
    --best_predictions "$G1_OUTPUT/predictions_best_fusion.json"

echo "[5/10] Tuning exact metadata reranking"
"$G1_PYTHON" -u -m retrieval.tune_exact_by_specificity \
    --predictions "$G1_OUTPUT/predictions_best_fusion.json" \
    --ground_truth "$G1_OUTPUT/ground_truth.json" --split test --catalog_size 47071 \
    --output "$G1_OUTPUT/results_exact_specificity_tuning.json" \
    --best_predictions "$G1_OUTPUT/predictions_exact_specificity_best.json"

echo "[6/10] Tuning initial diversity reranking"
"$G1_PYTHON" -u -m retrieval.tune_diversity_reranking \
    --predictions "$G1_OUTPUT/predictions_exact_specificity_best.json" \
    --ground_truth "$G1_OUTPUT/ground_truth.json" --catalog_size 47071 \
    --output "$G1_OUTPUT/results_diversity_tuning.json" \
    --best_predictions "$G1_OUTPUT/predictions_diversity_best.json"

echo "[7/10] Generating the adaptive top-100 reservoir (GPU query encoding)"
"$G1_PYTHON" -u -m retrieval.run_adaptive_goal_top100 \
    --split test --topk 100 --candidate_k 100 --dense_weight 0.25 --rrf_k 60 \
    --output "$G1_OUTPUT/predictions_adaptive_goal_top100_full.json"

echo "[8/10] Final selective-expansion fine search"
"$G1_PYTHON" -u -m retrieval.tune_selective_expansion \
    --predictions "$G1_OUTPUT/predictions_diversity_best.json" \
    --reservoir "$G1_OUTPUT/predictions_adaptive_goal_top100_full.json" \
    --ground_truth "$G1_OUTPUT/ground_truth.json" --catalog_size 47071 \
    --output "$G1_OUTPUT/results_selective_expansion_fine.json" \
    --best_predictions "$G1_OUTPUT/predictions_selective_expansion_fine_best.json"

echo "[9/10] Evaluating the new final predictions"
"$G1_PYTHON" -u -m retrieval.evaluation.evaluate \
    --predictions "$G1_OUTPUT/predictions_selective_expansion_fine_best.json" \
    --ground_truth "$G1_OUTPUT/ground_truth.json" --catalog_size 47071 \
    --output "$G1_OUTPUT/results_final.json"

echo "[10/10] Verifying saved rerankings and exporting the runtime artifact"
"$G1_PYTHON" -u -m retrieval.export_final \
    --artifacts-dir "$G1_OUTPUT" \
    --fusion-results results_prediction_fusion.json \
    --output "$G1_OUTPUT/final_retrieval.json"

echo "Completed. All generated files and pipeline.log are in $G1_OUTPUT"
echo "Next, in this project and Python environment:"
printf 'export MUSICCRS_RETRIEVAL_ARTIFACT=%q\n' "$G1_OUTPUT/final_retrieval.json"
echo 'python -m tests.full_music_smoke --quiet'
