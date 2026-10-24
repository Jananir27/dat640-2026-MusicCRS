"""Develop on separate queries, freeze, then evaluate without changing old files.

Development uses explicitly provisional metadata proxies. Evaluation reuses the
existing fixed AI labels and leaves new candidates unjudged for explicit review.
Neither set is independent human ground truth.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

from musiccrs.adaptive_retrieval import AdaptiveConfig, AdaptiveMusicRetriever
from musiccrs.relevance_feedback import name_key, values
from .metrics import graded_retrieval_metrics, latency_summary
from .part4 import prepare_query, feedback_for


PROFILES = {"conservative": AdaptiveConfig(influence=0.35),
            "balanced": AdaptiveConfig(influence=0.6), "responsive": AdaptiveConfig(influence=0.8)}
ROOT = Path("evaluation/part4")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def new_directory(path):
    path = Path(path)
    if path.exists():
        raise ValueError("Choose a new output directory; previous experiments must be preserved")
    path.mkdir(parents=True)
    return path


def load_system():
    import torch
    from retrieval.data_loader import MusicCatalogLoader
    from retrieval.bm25 import BM25Retriever
    from retrieval.dense import DenseRetriever
    torch.set_num_threads(4)
    catalog = MusicCatalogLoader()
    lexical = BM25Retriever(catalog=catalog, corpus_types=["track_name", "artist_name", "album_name", "tag_list"])
    dense = DenseRetriever()
    dense.encode_query("music")
    return catalog, lexical, dense


def run_query(retriever, query, mode, method, depth=10, repeats=1):
    times = []
    state = feedback_for(query)
    for _ in range(repeats):
        started = perf_counter()
        result = retriever.retrieve(query["search_text"], state, request=query["query"], mode=mode, topk=depth)
        times.append((perf_counter() - started) * 1000)
    row = {"query_id": query["id"], "category": query.get("category", "mixed_feedback"), "method": method,
           "predicted_track_ids": [t.track_id for t in result.tracks], "latency_ms": times,
           "original_candidate_ids": result.original_candidate_ids, "candidate_ids": result.candidate_ids,
           "new_candidate_ids": result.new_candidate_ids,
           "scores": {t.track_id: t.scores for t in result.tracks},
           "evidence": {t.track_id: t.evidence for t in result.tracks},
           "final_excluded_count": sum(state.excludes(t.metadata) for t in result.tracks)}
    print(query["id"], method, "new", len(result.new_candidate_ids), "top5",
          [t.metadata.get("track_name") for t in result.tracks[:5]], flush=True)
    return row


def pool_rows(queries, runs, catalog, prior=()):
    lookup = {q["id"]: q for q in queries}
    rows = {(r["query_id"], r["track_id"]): dict(r) for r in prior}
    for run in runs:
        q = lookup[run["query_id"]]
        for t in run["predicted_track_ids"]:
            key = (q["id"], t)
            if key in rows:
                continue
            metadata = catalog.metadata_dict[t]
            rows[key] = {"query_id": q["id"], "track_id": t, "query": q["query"],
                         "context": json.dumps({f: q.get(f, []) for f in
                             ("playlist", "likes", "dislikes", "excluded_artists", "excluded_tracks")}),
                         "title": ", ".join(values(metadata.get("track_name"))),
                         "artist": ", ".join(values(metadata.get("artist_name"))),
                         "album": ", ".join(values(metadata.get("album_name"))),
                         "tags": ", ".join(values(metadata.get("tag_list"))),
                         "grade": "", "reason": "", "assessor": ""}
    fields = ["query_id", "track_id", "query", "context", "title", "artist", "album", "tags", "grade", "reason", "assessor"]
    return [{f: row[f] for f in fields} for _, row in sorted(rows.items())]


def write_rows(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def development_grade(query, track, catalog):
    """Independent raw-tag development proxy; never uses rank, cosine or scores.

    This is deliberately not used to label the evaluation candidates. Human
    assessment is still required; it is not a musical/audio ground truth.
    """
    tags = {name_key(v) for v in values(track.get("tag_list"))}
    genre_words = {
        "rock": {"rock", "alternative rock", "hard rock", "indie rock", "grunge", "post grunge"},
        "alternative": {"alternative", "alternative rock"},
        "metal": {"metal", "nu metal", "alternative metal", "heavy metal", "progressive metal"},
        "hip hop": {"hip hop", "hip hop rap", "rap", "hiphop", "trap"},
        "pop": {"pop", "indie pop", "pop rock", "chamber pop", "folk pop"},
        "soul": {"soul", "r b soul", "neo soul"},
        "electronic": {"electronic", "electronica", "trip hop", "house", "techno", "dance"},
        "dance": {"dance", "house", "electronic", "techno"},
        "jazz": {"jazz", "cool jazz", "modal jazz", "jazz vocal", "vocal jazz", "bebop"},
    }
    genre_fit = any(tags & genre_words.get(g, {name_key(g)}) for g in query.get("genres", []))
    mood_words = {"chill": {"chill", "chillout", "mellow", "laid back", "calm"},
                  "energetic": {"energetic", "energy", "high energy"}}
    mood_fit = all(tags & mood_words.get(m, {name_key(m)}) for m in query.get("moods", []))
    seed_rows = [catalog.metadata_dict[t] for t in query.get("liked_ids", [])]
    artists = {name_key(a) for a in values(track.get("artist_name"))}
    same_artist = any(artists & {name_key(a) for a in values(s.get("artist_name"))} for s in seed_rows)
    specific = {"grunge", "post grunge", "nu metal", "alternative metal", "trip hop", "downtempo",
                "soul", "vocal jazz", "jazz vocal", "cool jazz", "bebop", "conscious hip hop", "trap", "house", "big beat"}
    shared = set().union(*(tags & {name_key(v) for v in values(s.get("tag_list"))} & specific for s in seed_rows))
    grade = 2 if genre_fit and mood_fit and (same_artist or shared) else 1 if genre_fit else 0
    return grade, f"Development-only raw-tag proxy: genre={genre_fit}, mood={mood_fit}, credited seed artist={same_artist}, shared specific tags={sorted(shared)}"


def metrics(document, rows):
    judgments = defaultdict(dict)
    assessors = set()
    for row in rows:
        if row["grade"] not in ("0", "1", "2") or not row["assessor"].strip():
            raise ValueError("All pooled candidates require explicit grades and assessor provenance")
        if row["track_id"] in judgments[row["query_id"]]:
            raise ValueError("Duplicate judgment")
        judgments[row["query_id"]][row["track_id"]] = int(row["grade"])
        assessors.add(row["assessor"])
    groups, per_query = defaultdict(list), {}
    for run in document["runs"]:
        result = graded_retrieval_metrics(run["predicted_track_ids"], judgments[run["query_id"]])
        per_query[run["query_id"], run["method"]] = result
        groups[run["method"]].append((run, result))
    methods = {}
    feedback_ids = [q["id"] for q in document["queries"] if q.get("liked_ids") or q.get("disliked_ids")]
    for method, items in groups.items():
        methods[method] = {metric: sum(r[metric] for _, r in items) / len(items)
                           for metric in ("ndcg@5", "precision@5")}
        methods[method]["excluded_tracks_returned"] = sum(run["final_excluded_count"] for run, _ in items)
        methods[method]["latency"] = latency_summary([t for run, _ in items for t in run["latency_ms"]])
        feedback = [r for run, r in items if run["query_id"] in feedback_ids]
        methods[method]["feedback_queries"] = {metric: sum(r[metric] for r in feedback) / len(feedback)
                                                for metric in ("ndcg@5", "precision@5")} if feedback else {}
    paired = {}
    for before, after in document.get("comparisons", []):
        paired[after + " minus " + before] = {
            metric: sum(per_query[q, after][metric] - per_query[q, before][metric] for q in feedback_ids) / len(feedback_ids)
            for metric in ("ndcg@5", "precision@5")}
    return {"judgment_status": "AI METADATA ASSESSMENT — not independent human or audio judgments",
            "query_count": len(document["queries"]), "feedback_query_count": len(feedback_ids),
            "methods": methods, "paired_feedback_delta": paired,
            "per_query": {q: {m: per_query[q, m] for m in methods} for q in judgments},
            "assessors": sorted(assessors)}


def develop(output_dir):
    output = new_directory(output_dir)
    development = json.loads((ROOT / "adaptive_development_queries.json").read_text())
    evaluation = json.loads((ROOT / "tobi_queries.json").read_text())
    assert {q["id"] for q in development}.isdisjoint(q["id"] for q in evaluation)
    def seed_names(queries):
        return {name_key(t["artist"]) for q in queries for f in ("likes", "dislikes", "playlist") for t in q.get(f, [])}
    assert seed_names(development).isdisjoint(seed_names(evaluation))
    save(output / "protocol.json", {"development_queries_sha256": digest(ROOT / "adaptive_development_queries.json"),
                                    "profiles": {name: asdict(config) for name, config in PROFILES.items()},
                                    "selection": "Highest development macro nDCG@5, then Precision@5; tie prefers earlier, weaker profile",
                                    "evaluation_labels_read": False,
                                    "caveat": "Development labels are metadata proxies; existing evaluation has previously been inspected, so it is not an unseen holdout."})
    catalog, lexical, dense = load_system()
    queries = [prepare_query(q, catalog) for q in development]
    retrievers = {name: AdaptiveMusicRetriever(catalog, lexical, dense, config=config) for name, config in PROFILES.items()}
    runs = []
    for query in queries:
        runs.append(run_query(retrievers["conservative"], query, "baseline", "baseline"))
        for name, retriever in retrievers.items():
            runs.append(run_query(retriever, query, "feedback_retrieval", name))
    document = {"queries": queries, "runs": runs,
                "comparisons": [("baseline", name) for name in PROFILES]}
    save(output / "runs.json", document)
    rows = pool_rows(queries, runs, catalog)
    lookup = {q["id"]: q for q in queries}
    for row in rows:
        grade, reason = development_grade(lookup[row["query_id"]], catalog.metadata_dict[row["track_id"]], catalog)
        row.update(grade=str(grade), reason=reason, assessor="development-only:metadata-proxy:REVIEW-REQUIRED")
    write_rows(output / "judgments.csv", rows)
    result = metrics(document, rows)
    save(output / "metrics.json", result)
    best = max(PROFILES, key=lambda name: tuple(result["methods"][name][m] for m in ("ndcg@5", "precision@5")))
    save(output / "selected_config.json", {"profile": best, "config": asdict(PROFILES[best]),
                                           "development_protocol_sha256": digest(output / "protocol.json"),
                                           "development_runs_sha256": digest(output / "runs.json"),
                                           "development_metrics": result["methods"][best],
                                           "frozen_before_evaluation": True})
    print("SELECTED", best, json.dumps(result["methods"], indent=2), flush=True)


def evaluate(config_path, output_dir, repeats):
    output = new_directory(output_dir)
    config = json.loads(Path(config_path).read_text())
    save(output / "protocol.json", {"selected_config_sha256": digest(config_path), "selected": config,
                                    "queries_sha256": digest(ROOT / "tobi_queries.json"),
                                    "old_judgments_sha256": digest(ROOT / "results/reviewed_judgments.csv"),
                                    "old_runs_sha256": digest(ROOT / "results/runs.json"),
                                    "repeats": repeats, "pool_depth": 10,
                                    "latency_scope": "New methods: warm retrieval/filtering/ranking; no explanations or transport. Old timing scope differs; do not claim a speed comparison."})
    catalog, lexical, dense = load_system()
    retriever = AdaptiveMusicRetriever(catalog, lexical, dense, config=AdaptiveConfig(**config["config"]))
    queries = [prepare_query(q, catalog) for q in json.loads((ROOT / "tobi_queries.json").read_text())]
    runs = []
    for query in queries:
        for mode in ("baseline", "rerank_only", "feedback_retrieval"):
            runs.append(run_query(retriever, query, mode, "adaptive_" + mode, repeats=repeats))
    # Freeze rankings BEFORE consulting prior relevance grades.
    save(output / "new_runs.json", {"queries": queries, "runs": runs})
    original = json.loads((ROOT / "results/runs.json").read_text())
    merged = {"queries": queries, "runs": original["runs"] + runs,
              "comparisons": [("hybrid", "hybrid_feedback"), ("adaptive_baseline", "adaptive_rerank_only"),
                              ("adaptive_baseline", "adaptive_feedback_retrieval"),
                              ("adaptive_rerank_only", "adaptive_feedback_retrieval"),
                              ("hybrid_feedback", "adaptive_feedback_retrieval")]}
    save(output / "runs.json", merged)
    with (ROOT / "results/reviewed_judgments.csv").open(newline="") as stream:
        prior = list(csv.DictReader(stream))
    rows = pool_rows(queries, runs, catalog, prior)
    write_rows(output / "judgments.csv", rows)
    save(output / "coverage.json", {"pooled_pairs": len(rows), "preserved_previous_labels": len(prior),
                                    "new_unjudged_pairs": sum(not r["grade"] for r in rows)})
    print("NEW CANDIDATES NEED REVIEW", sum(not r["grade"] for r in rows), flush=True)


def review_evaluation(directory):
    """Materialize explicit new-label decisions; retain every prior judgment."""
    import sqlite3
    directory = Path(directory)
    decision_path = directory / "review_decisions.json"
    decisions = json.loads(decision_path.read_text())
    for filename, expected in decisions["input_sha256"].items():
        if digest(directory / filename) != expected:
            raise ValueError("Frozen review input changed: " + filename)
    with (directory / "judgments.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    unknown = [r for r in rows if not r["grade"]]
    unique = {r["track_id"]: r for r in unknown}
    ordered = sorted(unique.values(), key=lambda r: (r["artist"].casefold(), r["title"].casefold(), r["track_id"]))
    track_keys = {f"A{i:03}": r for i, r in enumerate(ordered, 1)}
    new_labels = {}
    for q, groups in decisions["query_decisions"].items():
        for group in groups:
            if type(group["grade"]) is not int or group["grade"] not in (0, 1, 2) or not group["reason"].strip():
                raise ValueError("Invalid review decision")
            for key in group["keys"].split():
                pair = q, track_keys[key]["track_id"]
                if pair in new_labels:
                    raise ValueError("Duplicate new review decision")
                new_labels[pair] = key, group
    if set(new_labels) != {(r["query_id"], r["track_id"]) for r in unknown}:
        raise ValueError("Every new pooled candidate must be explicitly reviewed")
    with (ROOT / "results/reviewed_judgments.csv").open(newline="") as stream:
        prior = {(r["query_id"], r["track_id"]): r for r in csv.DictReader(stream)}
    document = json.loads((directory / "runs.json").read_text())
    queries = {q["id"]: q for q in document["queries"]}
    with sqlite3.connect(Path("cache/catalog.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as connection:
        catalog = {t: json.loads(m) for t, m in connection.execute("SELECT track_id,metadata_json FROM tracks")}
    revised, violations = [], []
    for original in rows:
        row = dict(original)
        pair = row["query_id"], row["track_id"]
        metadata = catalog[row["track_id"]]
        for field, source in (("title", "track_name"), ("artist", "artist_name"), ("album", "album_name"), ("tags", "tag_list")):
            if row[field] != ", ".join(values(metadata.get(source))):
                raise ValueError("Pool/catalog mismatch: " + str(pair))
        if pair in new_labels:
            key, group = new_labels[pair]
            caveat = decisions["caveats"].get(key, "")
            sources = decisions["external_verification"].get(key, [])
            row.update(grade=str(group["grade"]), reason=group["reason"] +
                       (" Caveat: " + caveat if caveat else "") +
                       (" Verified sources: " + "; ".join(sources) if sources else ""),
                       assessor=decisions["assessor"])
        elif pair not in prior or any(row[f] != prior[pair][f] for f in row):
            raise ValueError("A previously reviewed label was changed")
        state = feedback_for(queries[row["query_id"]])
        if state.excludes(metadata):
            violations.append(pair)
            if row["grade"] != "0":
                raise ValueError("Forbidden track received positive relevance")
        revised.append(row)
    if len(revised) != len({(r["query_id"], r["track_id"]) for r in revised}):
        raise ValueError("Duplicate pool identities")
    by_pair = {(r["query_id"], r["track_id"]): int(r["grade"]) for r in revised}
    checks = []
    for broad, specific, equal in (("tobi-g01", "tobi-e01", True), ("tobi-g03", "tobi-e02", True),
                                   ("tobi-e05", "tobi-f01", False), ("tobi-g03", "tobi-f02", False),
                                   ("tobi-e03", "tobi-f03", False), ("tobi-g01", "tobi-f04", False)):
        common = {t for q, t in by_pair if q == broad} & {t for q, t in by_pair if q == specific}
        for t in common:
            a, b = by_pair[broad, t], by_pair[specific, t]
            if (equal and a != b) or (not equal and b > a):
                raise ValueError("Related-query label inconsistency: " + str((broad, specific, t)))
        checks.append({"broad": broad, "specific": specific, "shared_candidates": len(common)})
    write_rows(directory / "reviewed_judgments.csv", revised)
    save(directory / "review_audit.json", {
        "preserved_old_pairs": len(prior), "new_reviewed_pairs": len(new_labels), "total_pairs": len(revised),
        "catalog_field_mismatches": 0, "hard_constraint_violations_in_pool": violations,
        "related_query_consistency_checks": checks,
        "decision_sha256": digest(decision_path), "reviewed_judgments_sha256": digest(directory / "reviewed_judgments.csv"),
        "track_key_map": {key: {f: row[f] for f in ("track_id", "title", "artist")} for key, row in track_keys.items()},
        "new_grade_counts": {str(g): sum(r["grade"] == str(g) and (r["query_id"], r["track_id"]) in new_labels for r in revised) for g in (0, 1, 2)},
        "external_verification": decisions["external_verification"],
        "human_review_required": True, "audio_review_performed": False})
    print("REVIEWED", len(new_labels), "new pairs; preserved", len(prior), "old labels")


def verify_replay(directory):
    """Check current implementation against all frozen new rankings and scores."""
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    frozen = json.loads((directory / "new_runs.json").read_text())
    catalog, lexical, dense = load_system()
    retriever = AdaptiveMusicRetriever(catalog, lexical, dense,
                                      config=AdaptiveConfig(**protocol["selected"]["config"]))
    lookup = {q["id"]: q for q in frozen["queries"]}
    mismatches, max_score_difference = [], 0.0
    for run in frozen["runs"]:
        query = lookup[run["query_id"]]
        state = feedback_for(query)
        mode = run["method"].removeprefix("adaptive_")
        result = retriever.retrieve(query["search_text"], state, request=query["query"], mode=mode, topk=10)
        ids = [t.track_id for t in result.tracks]
        if ids != run["predicted_track_ids"]:
            mismatches.append((query["id"], mode))
        for track in result.tracks:
            if track.track_id in run["scores"]:
                difference = abs(track.score - run["scores"][track.track_id]["final"])
                max_score_difference = max(max_score_difference, difference)
        print("REPLAY", query["id"], mode, "match", ids == run["predicted_track_ids"], flush=True)
    if mismatches or max_score_difference > 1e-6:
        raise ValueError(f"Replay differs: {mismatches}; largest score difference={max_score_difference}")
    save(directory / "replay_verification.json", {"rankings_verified": len(frozen["runs"]),
                                                "exact_top10_match": True, "maximum_score_difference": max_score_difference,
                                                "implementation_sha256": digest("musiccrs/adaptive_retrieval.py"),
                                                "baseline_adapter_sha256": digest("musiccrs/feedback_retrieval.py"),
                                                "frozen_new_runs_sha256": digest(directory / "new_runs.json"),
                                                "selected_config_sha256": protocol["selected_config_sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dev = sub.add_parser("develop")
    dev.add_argument("--output-dir", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--repeats", type=int, default=3)
    score = sub.add_parser("score")
    score.add_argument("--runs", required=True)
    score.add_argument("--judgments", required=True)
    score.add_argument("--output", required=True)
    review = sub.add_parser("review")
    review.add_argument("--directory", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--directory", required=True)
    args = parser.parse_args()
    if args.command == "develop":
        develop(args.output_dir)
    elif args.command == "evaluate":
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        evaluate(args.config, args.output_dir, args.repeats)
    elif args.command == "review":
        review_evaluation(args.directory)
    elif args.command == "verify":
        verify_replay(args.directory)
    else:
        with Path(args.judgments).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        result = metrics(json.loads(Path(args.runs).read_text()), rows)
        save(args.output, result)
        print(json.dumps({"methods": result["methods"], "paired_feedback_delta": result["paired_feedback_delta"]}, indent=2))


if __name__ == "__main__":
    main()
