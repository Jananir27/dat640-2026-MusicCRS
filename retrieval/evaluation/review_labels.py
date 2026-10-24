"""Validate and export the explicit Part 4 AI label review; never infer grades.

Replay: python -m retrieval.evaluation.review_labels
The decisions are in evaluation/part4/label_review_decisions.json. Changing the
frozen source files requires a new review, not silently reusing positional keys.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .part4 import score_benchmark


def read_csv(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def identity(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def as_list(value):
    return value if isinstance(value, list) else [value] if value else []


def same_song(left, right):
    return bool({identity(t) for t in as_list(left.get("track_name"))}
                & {identity(t) for t in as_list(right.get("track_name"))}) and bool(
                    {identity(a) for a in as_list(left.get("artist_name"))}
                    & {identity(a) for a in as_list(right.get("artist_name"))})


def markdown(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def review(root=Path("evaluation/part4"), catalog_path=Path("cache/catalog.sqlite3")):
    result_dir = root / "results"
    decision_path = root / "label_review_decisions.json"
    manifest = json.loads(decision_path.read_text())
    for filename, expected in manifest["input_sha256"].items():
        actual = hashlib.sha256((result_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen review input changed: {filename}; a new review is required")
    fields, rows = read_csv(result_dir / "provisional_judgments.csv")
    _, blind_rows = read_csv(result_dir / "judgments.csv")
    keyed = {(r["query_id"], r["track_id"]): r for r in rows}
    blind = {(r["query_id"], r["track_id"]): r for r in blind_rows}
    if len(keyed) != len(rows) or len(blind) != len(blind_rows) or keyed.keys() != blind.keys():
        raise ValueError("Duplicate or inconsistent pooled query/track identities")
    for key, row in keyed.items():
        if any(row[field] != blind[key][field] for field in fields if field not in {"grade", "reason", "assessor"}):
            raise ValueError(f"Draft and blind-pool metadata differ: {key}")
    document = json.loads((result_dir / "runs.json").read_text())
    queries = {q["id"]: q for q in document["queries"]}
    if set(queries) != set(manifest["query_decisions"]):
        raise ValueError("Review does not cover exactly the frozen query set")
    returned = {(run["query_id"], track) for run in document["runs"] for track in run["predicted_track_ids"]}
    if returned != keyed.keys():
        raise ValueError("Judgment pool does not match the union of frozen rankings")
    unique = {r["track_id"]: r for r in rows}
    ordered = sorted(unique.values(), key=lambda r: (r["artist"].casefold(), r["title"].casefold(), r["track_id"]))
    tracks = {f"T{i:03}": row for i, row in enumerate(ordered, 1)}
    keys_by_id = {row["track_id"]: key for key, row in tracks.items()}
    with sqlite3.connect(catalog_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        catalog = {track: json.loads(metadata) for track, metadata in connection.execute(
            "SELECT track_id, metadata_json FROM tracks")}
    for row in rows:
        metadata = catalog[row["track_id"]]
        for field, source in (("title", "track_name"), ("artist", "artist_name"),
                              ("album", "album_name"), ("tags", "tag_list")):
            if row[field] != ", ".join(str(v).strip() for v in as_list(metadata.get(source)) if str(v).strip()):
                raise ValueError(f"Pool/catalog metadata mismatch: {row['track_id']} {field}")
    decisions = {}
    for query_id, groups in manifest["query_decisions"].items():
        for group in groups:
            if type(group["grade"]) is not int or group["grade"] not in (0, 1, 2) or not group["reason"].strip():
                raise ValueError("Every explicit decision needs a valid grade and reason")
            for key in group["keys"].split():
                pair = (query_id, tracks[key]["track_id"])
                if pair in decisions:
                    raise ValueError(f"Duplicate review decision: {pair}")
                decisions[pair] = (key, group["grade"], group["reason"])
    if decisions.keys() != keyed.keys():
        raise ValueError("Explicit decisions must cover every pooled row, with no extras")
    if set(manifest["track_caveats"]) - tracks.keys():
        raise ValueError("Unknown track key in caveats")

    reviewed, constraint_violations = [], []
    for pair, original in keyed.items():
        key, grade, reason = decisions[pair]
        query, track = queries[pair[0]], catalog[pair[1]]
        forbidden_ids = set().union(*(set(query.get(field, [])) for field in
                                      ("playlist_ids", "liked_ids", "disliked_ids", "excluded_ids")))
        excluded_artists = {identity(a) for a in query.get("excluded_artists", [])}
        forbidden = (pair[1] in forbidden_ids
                     or any(same_song(track, catalog[t]) for t in forbidden_ids)
                     or bool(excluded_artists & {identity(a) for a in as_list(track.get("artist_name"))}))
        if forbidden:
            constraint_violations.append(list(pair))
            if grade != 0:
                raise ValueError(f"A hard-forbidden track received positive relevance: {pair}")
        caveat = manifest["track_caveats"].get(key, "")
        row = dict(original)
        # Full verbatim catalog tags remain in the tags column; text below is
        # the explicitly chosen query-specific rationale, never a classifier.
        row.update(grade=grade, reason=reason + (" Review caveat: " + caveat if caveat else ""),
                   assessor=manifest["assessor"], original_grade=original["grade"], track_key=key,
                   review_type=manifest["review_type"], review_caveat=caveat,
                   human_review_required="yes", hard_constraint_violation="yes" if forbidden else "no",
                   evidence_source="frozen pooled catalog metadata and resolved seed catalog records",
                   human_review_priority="high" if caveat or query["category"] in {
                       "artist_similarity", "playlist_context", "mixed_feedback"} else "standard")
        reviewed.append(row)
    consistency_checks = []
    # Identical requests differing only in an exclusion should agree on tracks
    # that do not violate it. More specific feedback must not receive stronger
    # relevance than the otherwise corresponding broad request.
    for broad_id, specific_id, relation in (
        ("tobi-g01", "tobi-e01", "equal"), ("tobi-g03", "tobi-e02", "equal"),
        ("tobi-e05", "tobi-f01", "no_higher"), ("tobi-g03", "tobi-f02", "no_higher"),
        ("tobi-e03", "tobi-f03", "no_higher"), ("tobi-g01", "tobi-f04", "no_higher"),
    ):
        broad = {r["track_id"]: r for r in reviewed if r["query_id"] == broad_id}
        specific = {r["track_id"]: r for r in reviewed if r["query_id"] == specific_id}
        common = broad.keys() & specific.keys()
        for track in common:
            if broad[track]["hard_constraint_violation"] == "yes" or specific[track]["hard_constraint_violation"] == "yes":
                continue
            a, b = broad[track]["grade"], specific[track]["grade"]
            if (relation == "equal" and a != b) or (relation == "no_higher" and b > a):
                raise ValueError(f"Inconsistent related-query grades: {broad_id} {specific_id} {track}")
        consistency_checks.append({"broad_query": broad_id, "specific_query": specific_id,
                                   "relation": relation, "shared_tracks_checked": len(common)})
    extra_fields = [field for field in reviewed[0] if field not in fields]
    output_path = result_dir / "reviewed_judgments.csv"
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + extra_fields)
        writer.writeheader()
        writer.writerows(reviewed)
    metrics = score_benchmark(result_dir / "runs.json", output_path)
    metrics["label_review"] = {
        "decision_file": str(decision_path), "reviewed_rows": len(reviewed),
        "changed_grades": sum(str(r["grade"]) != r["original_grade"] for r in reviewed),
        "human_review_required": True, "audio_review_performed": False,
        "independent_blind_assessment": False,
        "latency_note": "Reuses the original frozen retrieval timings; only labels and relevance metrics changed."
    }
    (result_dir / "reviewed_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    per_query = {}
    for query_id in queries:
        subset = [r for r in reviewed if r["query_id"] == query_id]
        per_query[query_id] = {"rows": len(subset),
                               "changes": sum(str(r["grade"]) != r["original_grade"] for r in subset),
                               "grades": dict(Counter(str(r["grade"]) for r in subset))}
    seed_ids = {t for q in queries.values() for field in
                ("playlist_ids", "liked_ids", "disliked_ids", "excluded_ids") for t in q.get(field, [])}
    audit = {"assessor": manifest["assessor"], "review_type": manifest["review_type"],
             "input_sha256": manifest["input_sha256"],
             "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
             "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
             "rows_reviewed": len(reviewed), "unique_tracks_reviewed": len(tracks),
             "query_count": len(queries), "catalog_field_mismatches": 0,
             "pool_coverage_verified": True, "duplicate_decisions": 0,
             "related_query_consistency_checks": consistency_checks,
             "hard_constraint_violations_in_pool": constraint_violations,
             "changed_grades": metrics["label_review"]["changed_grades"],
             "original_grade_counts": dict(Counter(r["original_grade"] for r in reviewed)),
             "reviewed_grade_counts": dict(Counter(str(r["grade"]) for r in reviewed)),
             "grade_transitions": dict(sorted(Counter(
                 f"{r['original_grade']}->{r['grade']}" for r in reviewed).items())),
             "per_query": per_query,
             "track_key_map": {key: {field: row[field] for field in ("track_id", "title", "artist")}
                               for key, row in tracks.items()},
             "seed_metadata": {t: catalog[t] for t in sorted(seed_ids)},
             "limitations": manifest["interpretation"][-2:]}
    (result_dir / "label_review_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    lines = ["# Part 4 relevance-label review: row-by-row decisions", "",
             "All 502 query/track pairs were explicitly reviewed by Codex using catalog metadata. "
             "These are AI judgments requiring human confirmation; no listening or independent blind assessment occurred.", "",
             "Full tags, context and original grades are preserved in `results/reviewed_judgments.csv`. "
             "Track keys map to UUIDs in `results/label_review_audit.json`. Grades are chosen in "
             "`label_review_decisions.json`; the export script only validates and materializes those choices.", ""]
    for query_id, query in queries.items():
        lines.extend([f"## {query_id}: {query['query']}", "",
                      "Context: `" + json.dumps({f: query.get(f, []) for f in
                          ("playlist", "likes", "dislikes", "excluded_artists", "excluded_tracks")}) + "`", "",
                      "| Track | Artist | Draft → reviewed | Rationale |", "| --- | --- | --- | --- |"])
        for row in (r for r in reviewed if r["query_id"] == query_id):
            lines.append("| " + " | ".join(markdown(v) for v in (
                f"{row['track_key']}: {row['title']}", row["artist"],
                f"{row['original_grade']} → {row['grade']}", row["reason"])) + " |")
        lines.append("")
    (root / "LABEL_REVIEW_DETAILS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows_reviewed": len(reviewed), "changed_grades": audit["changed_grades"],
                      "grade_transitions": audit["grade_transitions"],
                      "judgment_status": metrics["judgment_status"], "methods": metrics["methods"],
                      "paired_feedback_delta": metrics["paired_feedback_delta"]}, indent=2))
    return audit, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("evaluation/part4"))
    parser.add_argument("--catalog", type=Path, default=Path("cache/catalog.sqlite3"))
    args = parser.parse_args()
    review(args.root, args.catalog)


if __name__ == "__main__":
    main()
