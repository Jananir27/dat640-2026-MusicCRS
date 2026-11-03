"""Evaluate the actual standard MusicIntelligence/MusicService R1/R2 path.

All methods use identical structured constraints and fresh per-request state.
No external LLM or relevance-label generation is involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from musiccrs.music_intelligence import MusicIntelligence
from musiccrs.music_service import ConversationState, MusicService
from musiccrs.relevance_feedback import values
from musiccrs.request_constraints import resolve_exclusions
from .metrics import latency_summary
from .part4 import prepare_query, feedback_for, score_benchmark


METHODS = ("bm25", "semantic", "hybrid", "hybrid_feedback")
FIELDS = ("query_id", "track_id", "query", "context", "title", "artist", "album", "tags",
          "grade", "reason", "assessor")


def implementation_hashes():
    """Fingerprint project code, excluding nested environments and libraries."""
    excluded = {"venv", ".venv", "env", ".env", "site-packages", "node_modules", "__pycache__", ".git"}
    sources = []
    for root in ("musiccrs", "retrieval"):
        for directory, children, files in os.walk(root):
            children[:] = [child for child in children if child not in excluded]
            sources.extend(Path(directory) / name for name in files if name.endswith(".py"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}


def load_queries(paths):
    queries = [query for path in paths for query in json.loads(Path(path).read_text())]
    if not queries or len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Evaluation queries must be nonempty and have unique IDs")
    return queries


def hard_feedback(query, catalog):
    feedback = feedback_for(query)
    # Known songs and explicit bans have the same recording/artist identity
    # semantics for every method, including duplicate IDs and remasters.
    for group in ("playlist", "likes", "dislikes", "excluded_tracks"):
        for ref in query.get(group, []):
            resolved = resolve_exclusions(f"{ref['title']} by {ref['artist']}", catalog.metadata_dict, kind="track")
            feedback.excluded_track_ids.update(resolved.track_ids)
    return feedback


def run_integrated(queries, catalog, services, output_dir, *, repeats=3, judgment_sources=()):
    if type(repeats) is not int or repeats < 1 or set(services) != set(METHODS):
        raise ValueError("Provide all four methods and a positive integer repeat count")
    if not queries or len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Queries require unique IDs")
    resolved = [prepare_query(q, catalog) for q in queries]
    contexts = {q["id"]: hard_feedback(q, catalog) for q in resolved}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    runs, pool = [], {}
    for query in resolved:
        request = query.get("retrieval_query", query["query"])
        if not MusicIntelligence._is_recommendation_request(request):
            request = "Recommend " + request
        for method in METHODS:
            durations, previous = [], None
            for _ in range(repeats):
                feedback = deepcopy(contexts[query["id"]])
                if method != "hybrid_feedback":
                    feedback.votes.clear()  # already-known songs stay hard excluded
                    feedback.liked_artists.clear()
                state = ConversationState(playlist_ids=query["playlist_ids"][:], feedback=feedback)
                chat = MusicIntelligence(services[method])
                started = perf_counter()
                response, _ = chat.handle(request, state)
                durations.append((perf_counter() - started) * 1000)
                predicted = state.last_recommendation_ids[:]
                if previous is not None and previous != predicted:
                    raise ValueError(f"Nondeterministic repeated ranking for {query['id']} / {method}")
                previous = predicted
                if len(predicted) > 5 or len(predicted) != len(set(predicted)):
                    raise ValueError("The application returned an invalid top-five ranking")
                forbidden = sum(state.feedback.excludes(catalog.metadata_dict[t]) or t in state.playlist_ids
                                for t in predicted)
                if forbidden:
                    raise ValueError("The integrated application returned an excluded track")
            runs.append({"query_id": query["id"], "category": query["category"], "method": method,
                         "application_request": request, "application_response": response,
                         "predicted_track_ids": predicted, "latency_ms": durations,
                         "final_excluded_count": forbidden,
                         "ranking_evidence": state.ranking_evidence})
            for t in predicted:
                row = catalog.metadata_dict[t]
                pool[query["id"], t] = {
                    "query_id": query["id"], "track_id": t, "query": query["query"],
                    "context": json.dumps({key: query.get(key, []) for key in
                                           ("playlist", "likes", "dislikes", "excluded_artists", "excluded_tracks")}),
                    **{field: ", ".join(values(row.get(source))) for field, source in
                       (("title", "track_name"), ("artist", "artist_name"), ("album", "album_name"), ("tags", "tag_list"))},
                    "grade": "", "reason": "", "assessor": "",
                }
            print(f"INTEGRATED {query['id']} {method}: {len(predicted)} tracks", flush=True)
    sources = ["musiccrs/music_intelligence.py", "musiccrs/music_service.py", "musiccrs/feedback_workflow.py",
               "musiccrs/feedback_language.py", "musiccrs/request_constraints.py", "musiccrs/relevance_feedback.py",
               "musiccrs/adaptive_retrieval.py", "retrieval/final.py", "retrieval/hybrid.py", "retrieval/dense.py",
               "retrieval/evaluation/integrated.py"]
    document = {"schema_version": 2, "queries": resolved, "runs": runs, "repeats": repeats,
                "catalog_size": len(catalog.metadata_dict), "pool_depth": 5,
                "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
                "application_route": "standard MusicIntelligence.handle -> MusicService -> shared R2 adapter; no LLM",
                "method_definitions": {"bm25": "BM25 through the application service",
                                       "semantic": "Qwen dense through the application service",
                                       "hybrid": "FinalRetriever R1 path without soft feedback",
                                       "hybrid_feedback": "same FinalRetriever R1 path with the live R2 adapter"},
                "latency_scope": "Warm conversation handling, constraint extraction, retrieval, feedback, explanations and payload construction; excludes schema preparation, model/index initialization, transport and browser rendering"}
    (output / "runs.json").write_text(json.dumps(document, indent=2) + "\n")
    known, source_hashes = {}, {}
    for source in judgment_sources:
        p = Path(source)
        source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        with p.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("grade") not in ("0", "1", "2") or not row.get("assessor", "").strip():
                    continue
                key = row["query_id"], row["track_id"]
                if key in known and any(known[key].get(c) != row.get(c) for c in ("grade", "assessor")):
                    raise ValueError("Conflicting source judgments; reconcile explicitly before reuse")
                known[key] = row
    rows = [pool[key] for key in sorted(pool)]
    mismatch = 0
    for row in rows:
        old = known.get((row["query_id"], row["track_id"]))
        if old:
            same = (all(row[c] == old.get(c) for c in ("query", "title", "artist", "album", "tags"))
                    and json.loads(row["context"]) == json.loads(old.get("context", "{}")))
            if same:
                row.update({field: old.get(field, "") for field in ("grade", "reason", "assessor")})
            else:
                mismatch += 1
    for name, blank in (("judgments.csv", False), ("human_review_blank.csv", True)):
        with (output / name).open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, **({"grade": "", "reason": "", "assessor": ""} if blank else {})})
    coverage = {"query_count": len(resolved), "run_count": len(runs), "timing_samples": len(runs) * repeats,
                "pooled_pairs": len(rows), "reused_judgments": sum(bool(r["grade"]) for r in rows),
                "unjudged_pairs": sum(not r["grade"] for r in rows), "metadata_mismatches_not_reused": mismatch,
                "assessors": dict(Counter(r["assessor"] or "UNJUDGED" for r in rows)),
                "judgment_source_sha256": source_hashes, "excluded_tracks_returned": 0,
                "latency": {method: latency_summary([t for r in runs if r["method"] == method for t in r["latency_ms"]])
                            for method in METHODS},
                "scoring_note": "No new relevance labels generated. Complete judgments and human assessment remain required; use score only when all pooled grades are present."}
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    return document, coverage


def verify_integrated(directory, catalog, services):
    """Replay frozen application requests on current code; no relevance scoring."""
    source = Path(directory) / "runs.json"
    document = json.loads(source.read_text())
    queries = {q["id"]: q for q in document["queries"]}
    contexts = {qid: hard_feedback(q, catalog) for qid, q in queries.items()}
    mismatches, max_score_difference = [], 0.0
    for run in document["runs"]:
        query, method = queries[run["query_id"]], run["method"]
        feedback = deepcopy(contexts[query["id"]])
        if method != "hybrid_feedback":
            feedback.votes.clear()
            feedback.liked_artists.clear()
        state = ConversationState(playlist_ids=query["playlist_ids"][:], feedback=feedback)
        MusicIntelligence(services[method]).handle(run["application_request"], state)
        if state.last_recommendation_ids != run["predicted_track_ids"]:
            mismatches.append({"query_id": query["id"], "method": method,
                               "expected": run["predicted_track_ids"], "actual": state.last_recommendation_ids})
        for t in set(state.ranking_evidence) & set(run["ranking_evidence"]):
            old, current = run["ranking_evidence"][t]["scores"], state.ranking_evidence[t]["scores"]
            max_score_difference = max(max_score_difference,
                                       max((abs(current[k] - old[k]) for k in old.keys() & current.keys()), default=0.0))
        print(f"VERIFIED {query['id']} {method}", flush=True)
    return {"reference_runs_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "queries": len(queries), "rankings_replayed": len(document["runs"]),
            "matched": not mismatches, "mismatches": mismatches,
            "maximum_score_difference": max_score_difference,
            "current_source_sha256": implementation_hashes(),
            "scope": "Ranking replay only. Original warm latency samples belong to the recorded run; no new relevance labels or metric gains are claimed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--queries", nargs="+", default=["evaluation/part4/tobi_queries.json", "evaluation/part4/janani_25_queries.json"])
    run.add_argument("--output-dir", required=True)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--judgment-sources", nargs="*", default=[])
    score = sub.add_parser("score")
    score.add_argument("--directory", required=True)
    score.add_argument("--judgments", required=True)
    score.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--directory", required=True)
    verify.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "score":
        result = score_benchmark(Path(args.directory) / "runs.json", args.judgments)
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(json.dumps(result, indent=2))
        return
    from retrieval.data_loader import MusicCatalogLoader
    from retrieval.final import FinalRetriever
    started = perf_counter()
    catalog = MusicCatalogLoader()
    queries = load_queries(args.queries) if args.command == "run" else []
    [prepare_query(q, catalog) for q in queries]
    final = FinalRetriever(catalog=catalog)
    services = {"bm25": MusicService(catalog, final.hybrid.bm25), "semantic": MusicService(catalog, final.hybrid.dense),
                "hybrid": MusicService(catalog, final)}
    services["hybrid_feedback"] = services["hybrid"]
    final.encode_query("music")
    services["hybrid"]._feedback_adapter._adaptive_index(final, final)
    for service in services.values():
        service.get_r5_entities()
    setup_ms = (perf_counter() - started) * 1000
    if args.command == "verify":
        result = verify_integrated(args.directory, catalog, services)
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        if not result["matched"]:
            raise SystemExit("Recorded rankings changed; inspect the replay report.")
        print(f"PASS: replayed {result['rankings_replayed']} rankings")
        return
    _, coverage = run_integrated(queries, catalog, services, args.output_dir,
                                 repeats=args.repeats, judgment_sources=args.judgment_sources)
    (Path(args.output_dir) / "environment.json").write_text(json.dumps({
        "setup_and_warmup_ms": setup_ms, "embedding_model": "Qwen/Qwen3-Embedding-0.6B"}, indent=2) + "\n")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
