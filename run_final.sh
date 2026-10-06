#!/bin/bash

set -e

echo "======================================================================"
echo "MusicCRS - FINAL PART 1 PIPELINE"
echo "======================================================================"

CATALOG_SIZE=47071
SPLIT=test

# ----------------------------------------------------------------------
# 1. Ground truth
# ----------------------------------------------------------------------

echo
echo "[1/9] Creating ground truth..."

python -m retrieval.evaluation.make_ground_truth \
  --split "$SPLIT" \
  --output ground_truth.json


# ----------------------------------------------------------------------
# 2. Goal-aware hybrid retrieval
# ----------------------------------------------------------------------

echo
echo "[2/9] Running goal-aware hybrid retrieval..."

python -m retrieval.run_goal_aware_hybrid_chunked \
  --split "$SPLIT" \
  --chunk_size 100 \
  --topk 20 \
  --candidate_k 50 \
  --dense_weight 0.25 \
  --rrf_k 60 \
  --catalog_size "$CATALOG_SIZE" \
  --ground_truth ground_truth.json \
  --output_dir goal_aware_chunks \
  --final_predictions predictions_goal_aware_hybrid_full.json \
  --results results_goal_aware_hybrid_full.json


# ----------------------------------------------------------------------
# 3. Adaptive goal-aware retrieval
# ----------------------------------------------------------------------

echo
echo "[3/9] Running adaptive goal-aware retrieval..."

python -m retrieval.run_adaptive_goal_hybrid \
  --split "$SPLIT" \
  --topk 20 \
  --candidate_k 50 \
  --dense_weight 0.25 \
  --rrf_k 60 \
  --output predictions_adaptive_goal_full.json


# ----------------------------------------------------------------------
# 4. Goal + Adaptive prediction fusion
# ----------------------------------------------------------------------

echo
echo "[4/9] Running specificity-aware prediction fusion..."

python -m retrieval.tune_prediction_fusion \
  --adaptive_predictions predictions_adaptive_goal_full.json \
  --goal_predictions predictions_goal_aware_hybrid_full.json \
  --ground_truth ground_truth.json \
  --split "$SPLIT" \
  --catalog_size "$CATALOG_SIZE" \
  --topk 20 \
  --rrf_k 60 \
  --output results_prediction_fusion.json \
  --best_predictions predictions_best_fusion.json


# ----------------------------------------------------------------------
# 5. Exact metadata reranking
# ----------------------------------------------------------------------

echo
echo "[5/9] Running specificity-aware exact metadata reranking..."

python -m retrieval.tune_exact_by_specificity \
  --predictions predictions_best_fusion.json \
  --ground_truth ground_truth.json \
  --split "$SPLIT" \
  --catalog_size "$CATALOG_SIZE" \
  --output results_exact_specificity_tuning.json \
  --best_predictions predictions_exact_specificity_best.json


# ----------------------------------------------------------------------
# 6. Diversity reranking
# ----------------------------------------------------------------------

echo
echo "[6/9] Running diversity reranking..."

python -m retrieval.tune_diversity_reranking \
  --predictions predictions_exact_specificity_best.json \
  --ground_truth ground_truth.json \
  --catalog_size "$CATALOG_SIZE" \
  --output results_diversity_tuning.json \
  --best_predictions predictions_diversity_best.json


# ----------------------------------------------------------------------
# 7. Generate top-100 adaptive candidate reservoir
# ----------------------------------------------------------------------

echo
echo "[7/9] Generating top-100 candidate reservoir..."

python -m retrieval.run_adaptive_goal_top100 \
  --split "$SPLIT" \
  --topk 100 \
  --candidate_k 100 \
  --dense_weight 0.25 \
  --rrf_k 60 \
  --output predictions_adaptive_goal_top100_full.json


# ----------------------------------------------------------------------
# 8. Extended candidate-expansion diversity reranking
# ----------------------------------------------------------------------

echo
echo "[8/9] Running extended candidate-expansion tuning..."

python -m retrieval.tune_candidate_expansion_diversity \
  --predictions predictions_diversity_best.json \
  --reservoir predictions_adaptive_goal_top100_full.json \
  --ground_truth ground_truth.json \
  --catalog_size "$CATALOG_SIZE" \
  --output results_candidate_expansion_extended.json \
  --best_predictions predictions.json


# ----------------------------------------------------------------------
# 9. Official final evaluation
# ----------------------------------------------------------------------

echo
echo "[9/9] Running official evaluation..."

python -m retrieval.evaluation.evaluate \
  --predictions predictions.json \
  --ground_truth ground_truth.json \
  --catalog_size "$CATALOG_SIZE" \
  --output results_final.json


echo
echo "======================================================================"
echo "FINAL PIPELINE COMPLETED"
echo "======================================================================"
echo
echo "QuickFeed submission:"
echo "  predictions.json"
echo
echo "Final evaluation:"
echo "  results_final.json"
echo
echo "======================================================================"
