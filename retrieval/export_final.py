"""Freeze the saved G1 settings and statistics for online retrieval.

Run on the machine holding the G1 artifacts. Replays both diversity stages and
requires an exact match to the saved rankings before writing the runtime file.
This does not run a model, retune parameters, or read ground truth.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .final import DEFAULT_ARTIFACT, SELECTIVE_KEYS, validate_artifact
from .tune_diversity_reranking import rerank_prediction
from .tune_selective_expansion import selective_rerank


def prediction_map(rows, name):
    output = {}
    for row in rows:
        key = (row["session_id"], row["turn_number"])
        tracks = row["predicted_track_ids"]
        if key in output or not tracks or len(tracks) != len(set(tracks)):
            raise ValueError(f"Invalid/duplicate prediction in {name}: {key}")
        if not all(isinstance(track_id, str) for track_id in tracks):
            raise ValueError(f"Non-string track ID in {name}: {key}")
        output[key] = tracks
    if not output:
        raise ValueError(f"Empty predictions: {name}")
    return output


def statistics(rankings):
    frequency = Counter(track_id for tracks in rankings.values() for track_id in tracks)
    return {"total_turns": len(rankings), "frequency": dict(frequency)}


def export_artifact(directory, output, *, fusion_results=None, final_predictions=None,
                    candidate_k=50, dense_weight=0.25, rrf_k=60, fusion_rrf_k=60):
    directory = Path(directory)
    provenance = {}

    def read(name):
        path = Path(name)
        if not path.is_absolute():
            path = directory / path
        raw = path.read_bytes()
        provenance[path.name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    if fusion_results is None:
        # run_final.sh and the standalone tuning script use different names.
        choices = [name for name in ("results_prediction_fusion.json", "prediction_fusion_tuning_results.json")
                   if (directory / name).is_file()]
        if len(choices) != 1:
            raise ValueError("Specify --fusion-results: expected exactly one saved fusion result file.")
        fusion_results = choices[0]
    fusion = read(fusion_results)
    exact = read("results_exact_specificity_tuning.json")
    diversity = read("results_diversity_tuning.json")
    selective = read("results_selective_expansion_fine.json")
    exact_map = prediction_map(read("predictions_exact_specificity_best.json"), "exact predictions")
    base_map = prediction_map(read("predictions_diversity_best.json"), "diversity predictions")
    reservoir_map = prediction_map(read("predictions_adaptive_goal_top100_full.json"), "reservoir")
    final_map = prediction_map(read(final_predictions or "predictions_selective_expansion_fine_best.json"), "final predictions")
    if not (exact_map.keys() == base_map.keys() == reservoir_map.keys() == final_map.keys()):
        raise ValueError("G1 prediction files have different session/turn keys.")
    if any(len(tracks) > 20 for mapping in (exact_map, base_map, final_map) for tracks in mapping.values()):
        raise ValueError("Original/final predictions must contain at most 20 tracks per turn.")
    if any(len(tracks) > 100 for tracks in reservoir_map.values()):
        raise ValueError("Reservoir must contain at most 100 tracks per turn.")

    artifact = validate_artifact({
        "schema_version": 1,
        "hybrid": {"candidate_k": candidate_k, "reservoir_k": 100,
                   "dense_weight": dense_weight, "rrf_k": rrf_k, "fusion_rrf_k": fusion_rrf_k},
        "fusion_weights": fusion["best_weights"],
        "exact_weights": exact["best_weights"],
        "diversity": {key: diversity["best_configuration"][key]
                      for key in ("protect_top", "novelty_weight")},
        "selective": {key: selective["best_configuration"][key] for key in SELECTIVE_KEYS},
        "diversity_statistics": statistics(exact_map),
        "selective_statistics": statistics(base_map),
        "reported_metrics": selective["best_metrics"],
        "source_sha256": provenance,
    })
    for key, tracks in exact_map.items():
        stats = artifact["diversity_statistics"]
        base = rerank_prediction(tracks, stats["frequency"], stats["total_turns"], **artifact["diversity"])
        if base != base_map[key]:
            raise ValueError(f"Saved diversity configuration does not reproduce {key}.")
        stats = artifact["selective_statistics"]
        final = selective_rerank(base, reservoir_map[key], stats["frequency"], stats["total_turns"], **artifact["selective"])
        if final != final_map[key]:
            raise ValueError(f"Saved selective configuration does not reproduce {key}.")

    artifact["verified_rerank_turns"] = len(base_map)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--output", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--fusion-results")
    parser.add_argument("--final-predictions", help="Use predictions.json only if it is the final fine-search output")
    parser.add_argument("--candidate-k", type=int, default=50, help="Original top-20 branch depth from run_final.sh")
    parser.add_argument("--dense-weight", type=float, default=0.25)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--fusion-rrf-k", type=int, default=60)
    args = parser.parse_args()
    artifact = export_artifact(
        args.artifacts_dir, args.output, fusion_results=args.fusion_results,
        final_predictions=args.final_predictions, candidate_k=args.candidate_k,
        dense_weight=args.dense_weight, rrf_k=args.rrf_k, fusion_rrf_k=args.fusion_rrf_k,
    )
    print(f"Verified both rerankers on {artifact['verified_rerank_turns']} turns.")
    print(f"Saved runtime artifact: {args.output}")
    print("Hybrid settings follow run_final.sh defaults unless explicitly overridden;")
    print("fresh hybrid retrieval still needs validation on the target machine.")


if __name__ == "__main__":
    main()
