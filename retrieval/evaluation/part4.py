"""Part 4 pooled evaluation: run -> judge (or draft labels) -> score.

The 25 Tobi queries can be combined with Janani's 25 using the same schema.
Model initialization/warmup is outside timed requests; every timed hybrid and
feedback request includes retrieval, filtering, and (when applicable) encoding.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from time import perf_counter

from musiccrs.feedback_language import split_exclusion_request
from musiccrs.music_features import classify_tags, track_style
from musiccrs.recommendation_explanations import explain_recommendation
from musiccrs.relevance_feedback import FeedbackState, RankedTrack, RocchioFeedback, name_key, values
from .metrics import graded_retrieval_metrics, latency_summary


def resolve_track(reference, catalog):
    matches = [row for row in catalog.metadata_dict.values()
               if name_key(reference["title"]) in {name_key(t) for t in values(row.get("track_name"))}
               and name_key(reference["artist"]) in {name_key(a) for a in values(row.get("artist_name"))}]
    if not matches:
        raise ValueError(f"Evaluation reference absent from catalog: {reference}")
    return sorted(matches, key=lambda row: (-float(row.get("popularity") or 0), row["track_id"]))[0]["track_id"]


def prepare_query(query, catalog):
    prepared = dict(query)
    for source, target in (("playlist", "playlist_ids"), ("likes", "liked_ids"),
                           ("dislikes", "disliked_ids"), ("excluded_tracks", "excluded_ids")):
        prepared[target] = [resolve_track(ref, catalog) for ref in query.get(source, [])]
    text, _ = split_exclusion_request(query.get("retrieval_query", query["query"]))
    for track_id in prepared["playlist_ids"]:
        track = catalog.id_to_metadata(track_id)
        text += "; playlist seed: " + "; ".join(
            ", ".join(values(track.get(field))) for field in ("track_name", "artist_name", "tag_list"))
    prepared["search_text"] = text
    return prepared


def feedback_for(query):
    state = FeedbackState(excluded_artists={name_key(a) for a in query.get("excluded_artists", [])})
    # Already-known seeds and hard constraints are identical for all baselines.
    state.excluded_track_ids.update(query["playlist_ids"] + query["liked_ids"] + query["excluded_ids"])
    for track_id in query["liked_ids"]:
        state.rate(track_id, True)
    for track_id in query["disliked_ids"]:
        state.rate(track_id, False)
    return state


def run_benchmark(queries, catalog, lexical, dense, hybrid, output_dir, pool_depth=10, repeats=1):
    if pool_depth < 5 or repeats < 1:
        raise ValueError("Pool depth must be at least 5 and repeats at least 1")
    if not queries or len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Queries require unique IDs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = [prepare_query(q, catalog) for q in queries]
    runs, pool = [], {}
    depth = min(max(50, pool_depth * 3), len(catalog.metadata_dict))
    for query in resolved:
        state = feedback_for(query)
        text = query["search_text"]
        for method in ("bm25", "semantic", "hybrid", "hybrid_feedback"):
            times = []
            for _ in range(repeats):
                started = perf_counter()
                retriever = lexical if method == "bm25" else dense if method == "semantic" else hybrid
                raw = list(dict.fromkeys(retriever.text_to_item_retrieval(text, depth)))
                raw = [track_id for track_id in raw if track_id in catalog.metadata_dict]
                forbidden = sum(state.excludes(catalog.metadata_dict[t]) for t in raw)
                tracks = [catalog.metadata_dict[t] for t in raw if not state.excludes(catalog.metadata_dict[t])]
                ranking_evidence = {}
                if method == "hybrid_feedback" and state.votes:
                    embeddings = dense.get_track_embeddings([t["track_id"] for t in tracks] + list(state.votes))
                    candidates = [RankedTrack(t["track_id"], t, 1 - rank / max(1, len(tracks)))
                                  for rank, t in enumerate(tracks)]
                    ranked = RocchioFeedback().rerank(candidates, state, embeddings, dense.encode_query(text))
                    tracks = [t.metadata for t in ranked]
                    ranking_evidence = {t.track_id: t.evidence for t in ranked}
                result = [t["track_id"] for t in tracks[:pool_depth]]
                playlist = [catalog.metadata_dict[t] for t in query["playlist_ids"]]
                for track in tracks[:5]:
                    explain_recommendation(track, text, playlist,
                                           state if method == "hybrid_feedback" else None,
                                           ranking_evidence.get(track["track_id"]), catalog.metadata_dict)
                times.append((perf_counter() - started) * 1000)
            runs.append({"query_id": query["id"], "category": query["category"], "method": method,
                         "predicted_track_ids": result, "latency_ms": times,
                         "raw_excluded_count": forbidden,
                         "final_excluded_count": sum(state.excludes(catalog.metadata_dict[t]) for t in result)})
            for track_id in result:
                row = catalog.metadata_dict[track_id]
                pool[(query["id"], track_id)] = {
                    "query_id": query["id"], "track_id": track_id, "query": query["query"],
                    "context": json.dumps({key: query.get(key, []) for key in
                                           ("playlist", "likes", "dislikes", "excluded_artists", "excluded_tracks")}),
                    "title": ", ".join(values(row.get("track_name"))),
                    "artist": ", ".join(values(row.get("artist_name"))),
                    "album": ", ".join(values(row.get("album_name"))),
                    "tags": ", ".join(values(row.get("tag_list"))),
                    "grade": "", "reason": "", "assessor": "",
                }
            print(f"{query['id']} {method}: {len(result)} results, {times[-1]:.1f} ms", flush=True)
    document = {"schema_version": 1, "queries": resolved, "runs": runs,
                "catalog_size": len(catalog.metadata_dict),
                "retrievers": {"bm25": type(lexical).__name__, "semantic": type(dense).__name__,
                               "hybrid": type(hybrid).__name__},
                "feedback_parameters": {"alpha": 1.0, "beta": 0.75, "gamma": 0.25, "weight": 0.75},
                "latency_scope": "Warm retrieval, filtering, feedback and explanations; model initialization and transport excluded",
                "repeats": repeats}
    (output_dir / "runs.json").write_text(json.dumps(document, indent=2) + "\n")
    rows = [pool[key] for key in sorted(pool)]
    with (output_dir / "judgments.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["query_id", "track_id", "grade"])
        writer.writeheader()
        writer.writerows(rows)
    return document


def draft_judgments(run_path, pool_path, output_path):
    """Produce transparent metadata-only draft labels for assessor review.

    These are a proxy, not human judgments or an audio-based relevance study.
    They never use retrieval scores, Rocchio similarities, or system identity.
    """
    document = json.loads(Path(run_path).read_text())
    queries = {query["id"]: query for query in document["queries"]}
    with Path(pool_path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        query = queries[row["query_id"]]
        genres, moods = classify_tags(row["tags"].split(", "))
        styles = set(track_style({"tag_list": row["tags"].split(", ")}))
        wanted_genres, wanted_moods = set(query.get("genres", [])), set(query.get("moods", []))
        # Use catalog metadata to judge the requested content, not model outputs.
        artist_match = any(name_key(a) in {name_key(v) for v in row["artist"].split(", ")}
                           for a in query.get("artists", []))
        genre_match = bool(genres & wanted_genres)
        mood_match = bool(moods & wanted_moods)
        style_match = bool(styles & set(query.get("styles", [])))
        if artist_match or style_match or (genre_match and (not wanted_moods or mood_match)):
            grade = 2
        elif genre_match or mood_match:
            grade = 1
        else:
            grade = 0
        row.update(grade=grade, assessor="metadata-proxy:REVIEW-REQUIRED",
                   reason=f"Verified matches: genres={sorted(genres & wanted_genres)}, moods={sorted(moods & wanted_moods)}, styles={sorted(styles & set(query.get('styles', [])))}, artist={artist_match}")
    with Path(output_path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_benchmark(run_path, judgments_path):
    document = json.loads(Path(run_path).read_text())
    judgments = defaultdict(dict)
    provisional = False
    provenance = defaultdict(int)
    with Path(judgments_path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["track_id"] in judgments[row["query_id"]]:
                raise ValueError("Duplicate query/track judgment")
            if row["grade"] not in {"0", "1", "2"}:
                raise ValueError("Complete all pooled 0–2 judgments before scoring")
            judgments[row["query_id"]][row["track_id"]] = int(row["grade"])
            provisional |= "REVIEW-REQUIRED" in row.get("assessor", "") or not row.get("assessor")
            assessor = row.get("assessor", "")
            if assessor.startswith("codex-ai:metadata-reviewed:"):
                provenance["ai_metadata_review"] += 1
            elif "REVIEW-REQUIRED" in assessor or not assessor:
                provenance["unreviewed_or_proxy"] += 1
            else:
                provenance["named_assessor"] += 1
    per_query, groups = {}, defaultdict(list)
    for run in document["runs"]:
        scores = graded_retrieval_metrics(run["predicted_track_ids"], judgments[run["query_id"]])
        scores["final_excluded_count"] = run["final_excluded_count"]
        per_query[(run["query_id"], run["method"])] = scores
        groups[run["method"]].append((run, scores))
    methods, categories = {}, defaultdict(dict)
    for method, items in groups.items():
        methods[method] = {metric: sum(scores[metric] for _, scores in items) / len(items)
                           for metric in ("ndcg@5", "precision@5")}
        methods[method].update(latency_summary([time for run, _ in items for time in run["latency_ms"]]))
        methods[method]["excluded_tracks_returned"] = sum(scores["final_excluded_count"] for _, scores in items)
        for category in sorted({run["category"] for run, _ in items}):
            subset = [scores for run, scores in items if run["category"] == category]
            categories[category][method] = {metric: sum(s[metric] for s in subset) / len(subset)
                                            for metric in ("ndcg@5", "precision@5")}
    feedback_ids = [q["id"] for q in document["queries"] if q.get("liked_ids") or q.get("disliked_ids")]
    paired = {metric: sum(per_query[(q, "hybrid_feedback")][metric] - per_query[(q, "hybrid")][metric]
                         for q in feedback_ids) / max(1, len(feedback_ids))
              for metric in ("ndcg@5", "precision@5")}
    status = "PROVISIONAL METADATA PROXY — requires assessor review" if provisional else "assessor-labeled"
    if provenance["ai_metadata_review"]:
        status = ("AI-REVIEWED METADATA — requires independent human assessment"
                  if len([count for count in provenance.values() if count]) == 1
                  else "MIXED JUDGMENTS — includes AI-reviewed metadata; requires independent human assessment")
    return {"judgment_status": status,
            "judgment_provenance": {key: count for key, count in provenance.items() if count},
            "query_count": len(document["queries"]), "feedback_query_count": len(feedback_ids),
            "methods": methods, "by_category": dict(categories), "paired_feedback_delta": paired,
            "latency_scope": document["latency_scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--queries", default="evaluation/part4/tobi_queries.json")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--pool-depth", type=int, default=10)
    run.add_argument("--repeats", type=int, default=1)
    draft = sub.add_parser("draft-judgments")
    draft.add_argument("--runs", required=True)
    draft.add_argument("--pool", required=True)
    draft.add_argument("--output", required=True)
    score = sub.add_parser("score")
    score.add_argument("--runs", required=True)
    score.add_argument("--judgments", required=True)
    score.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        from retrieval.data_loader import MusicCatalogLoader
        from retrieval.bm25 import BM25Retriever
        from retrieval.dense import DenseRetriever
        from retrieval.hybrid import HybridRetriever
        started = perf_counter()
        catalog = MusicCatalogLoader()
        queries = json.loads(Path(args.queries).read_text())
        [prepare_query(query, catalog) for query in queries]  # validate before expensive model loading
        lexical = BM25Retriever(catalog=catalog, corpus_types=["track_name", "artist_name", "album_name", "tag_list"])
        dense = DenseRetriever()
        hybrid = HybridRetriever(bm25_retriever=lexical, dense_retriever=dense)
        dense.encode_query("music")  # warm model before timed requests
        setup_ms = (perf_counter() - started) * 1000
        document = run_benchmark(queries, catalog, lexical, dense, hybrid,
                                 args.output_dir, args.pool_depth, args.repeats)
        document["setup_and_warmup_ms"] = setup_ms
        (Path(args.output_dir) / "runs.json").write_text(json.dumps(document, indent=2) + "\n")
    elif args.command == "draft-judgments":
        draft_judgments(args.runs, args.pool, args.output)
    else:
        result = score_benchmark(args.runs, args.judgments)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
