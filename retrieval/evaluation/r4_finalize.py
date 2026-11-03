"""Import submitted human reviews and report frozen R4 results without inventing labels.

Inputs stay unchanged. Optional explicit AI decisions must be separately supplied
and are always identified as provisional. No retrieval scores are used as labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from .part4 import score_benchmark
from .metrics import latency_summary


EVIDENCE = ("query", "context", "title", "artist", "album", "tags")
FIELDS = ("query_id", "track_id", *EVIDENCE, "grade", "reason", "assessor")
METHODS = ("bm25", "semantic", "hybrid", "hybrid_feedback")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair(row):
    return row["query_id"], row["track_id"]


def read_rows(path):
    """Read a flat review CSV/XLSX. Never execute formulas or trust cached grades."""
    path = Path(path)
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif path.suffix.casefold() == ".xlsx":
        sheets = []
        with ZipFile(path) as archive:
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared = ["".join(t.text or "" for t in node.findall(".//s:t", NS)) for node in
                          ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
            for name in archive.namelist():
                if not (name.startswith("xl/worksheets/sheet") and name.endswith(".xml")):
                    continue
                grid = []
                formulas = False
                for row in ET.fromstring(archive.read(name)).findall(".//s:sheetData/s:row", NS):
                    cells = {}
                    for cell in row.findall("s:c", NS):
                        formulas |= cell.find("s:f", NS) is not None
                        column = "".join(c for c in cell.get("r", "") if c.isalpha())
                        value = cell.find("s:v", NS)
                        text = "" if value is None else value.text or ""
                        if cell.get("t") == "s":
                            text = shared[int(text)]
                        elif cell.get("t") == "inlineStr":
                            text = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
                        cells[column] = text
                    if any(cells.values()):
                        grid.append(cells)
                if grid and set(FIELDS) <= set(grid[0].values()):
                    if formulas:
                        raise ValueError("Review data must contain literal values, not formulas")
                    sheets.append([{field: row.get(col, "") for col, field in grid[0].items() if field}
                                   for row in grid[1:]])
            if len(sheets) != 1:
                raise ValueError("Expected exactly one flat review sheet with the standard column names")
            rows = sheets[0]
    else:
        raise ValueError("Reviews must be CSV or XLSX")
    if not rows or any(not set(FIELDS) <= row.keys() for row in rows):
        raise ValueError("Review input requires the standard query/track, evidence and assessment columns")
    if len({pair(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate query/track rows in review input")
    return rows


def canonical_review(row, source):
    """Accept exact evidence, JSON key ordering, or a reversible Mac Roman error."""
    fixed, repairs = dict(row), []
    for field in EVIDENCE:
        if row[field] == source[field]:
            continue
        if field == "context" and json.loads(row[field]) == json.loads(source[field]):
            fixed[field] = source[field]
            repairs.append({"field": field, "operation": "JSON serialization only"})
            continue
        try:
            repaired = row[field].encode("mac_roman").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            repaired = None
        if repaired != source[field]:
            raise ValueError(f"Review evidence differs from frozen pool: {pair(row)} / {field}")
        repairs.append({"field": field, "operation": "reversed UTF-8 decoded as Mac Roman",
                        "submitted": row[field], "canonical": source[field]})
        fixed[field] = source[field]
    if row["grade"] not in {"0", "1", "2"} or not row["reason"].strip() or not row["assessor"].strip():
        raise ValueError("Every submitted human grade needs literal 0–2, reason and assessor")
    assessor = row["assessor"].casefold()
    if assessor.startswith(("codex-ai:", "ai-assisted:", "ai:")) or any(
            flag in assessor for flag in ("proxy", "provisional", "review-required")):
        raise ValueError("AI/proxy labels cannot be imported as submitted human reviews")
    return fixed, repairs


def agreement(pairs):
    """Descriptive agreement with AI grades, not inter-human reliability."""
    confusion = [[0] * 3 for _ in range(3)]
    for ai, human in pairs:
        confusion[int(ai)][int(human)] += 1
    n = len(pairs)
    return {"pairs": n, "exact_matches": sum(confusion[i][i] for i in range(3)),
            "exact_agreement": sum(confusion[i][i] for i in range(3)) / n if n else None,
            "mean_absolute_grade_difference": sum(abs(int(a) - int(h)) for a, h in pairs) / n if n else None,
            "confusion_ai_rows_human_columns_0_1_2": confusion}


def score_rows(run_path, rows):
    # A temporary computational CSV lets the existing, tested common scorer
    # remain authoritative. Persisted review outputs are JSON with provenance.
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "judgments.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        document = json.loads(Path(run_path).read_text())
        if "latency_scope" not in document:
            document["latency_scope"] = "Not recorded in this historical run; do not compare its timings with the integrated benchmark"
            normalized = Path(temp) / "legacy_runs.json"
            normalized.write_text(json.dumps(document))
            run_path = normalized
        return score_benchmark(run_path, path)


def validate_pool(document, rows):
    keys = {pair(row) for row in rows}
    returned = {(run["query_id"], t) for run in document["runs"] for t in run["predicted_track_ids"]}
    if len(keys) != len(rows) or keys != returned:
        raise ValueError("Judgment pool must exactly match the frozen union of returned tracks")
    query_ids = {q["id"] for q in document["queries"]}
    if len(query_ids) != len(document["queries"]):
        raise ValueError("Frozen queries must have unique IDs")
    expected = {(q, method) for q in query_ids for method in METHODS}
    runs = {(r["query_id"], r["method"]) for r in document["runs"]}
    if len(runs) != len(document["runs"]) or runs != expected:
        raise ValueError("Every query requires exactly one ranking from each common method")
    for run in document["runs"]:
        tracks = run["predicted_track_ids"]
        if len(tracks) > 5 or len(set(tracks)) != len(tracks) or run["final_excluded_count"]:
            raise ValueError("Invalid ranking or exclusion violation in frozen run")


def independent_scores(document, rows, metrics):
    """Audit the scorer with standard-library math and expose paired outcomes."""
    judgments = {}
    for row in rows:
        judgments.setdefault(row["query_id"], {})[row["track_id"]] = int(row["grade"])
    per_query = {}
    for run in document["runs"]:
        grades = judgments[run["query_id"]]
        ranked = [grades[t] for t in run["predicted_track_ids"][:5]]
        weights = [1 / math.log2(rank + 1) for rank in range(1, 6)]
        ideal = sorted(grades.values(), reverse=True)[:5]
        numerator = math.fsum((2 ** grade - 1) * weight for grade, weight in zip(ranked, weights))
        denominator = math.fsum((2 ** grade - 1) * weight for grade, weight in zip(ideal, weights))
        per_query.setdefault(run["query_id"], {})[run["method"]] = {
            "ndcg@5": numerator / denominator if denominator else 0.0,
            "precision@5": len([grade for grade in ranked if grade > 0]) / 5}
    maximum_difference = 0.0
    for method in METHODS:
        for metric in ("ndcg@5", "precision@5"):
            average = math.fsum(q[method][metric] for q in per_query.values()) / len(per_query)
            maximum_difference = max(maximum_difference, abs(average - metrics["methods"][method][metric]))
    if maximum_difference > 1e-12:
        raise ValueError("Independent relevance metric calculation disagrees with the common scorer")
    feedback_ids = [q["id"] for q in document["queries"] if q.get("liked_ids") or q.get("disliked_ids")]
    runs = {(r["query_id"], r["method"]): r for r in document["runs"]}
    non_feedback_changes = [qid for qid in per_query if qid not in feedback_ids and
                            runs[qid, "hybrid"]["predicted_track_ids"] != runs[qid, "hybrid_feedback"]["predicted_track_ids"]]
    pairs = {qid: {metric: per_query[qid]["hybrid_feedback"][metric] - per_query[qid]["hybrid"][metric]
                   for metric in ("ndcg@5", "precision@5")} for qid in feedback_ids}
    outcomes = {metric: dict(Counter("improved" if d[metric] > 1e-12 else "worsened" if d[metric] < -1e-12 else "tied"
                                    for d in pairs.values())) for metric in ("ndcg@5", "precision@5")}
    for metric in ("ndcg@5", "precision@5"):
        observed = math.fsum(d[metric] for d in pairs.values()) / max(1, len(pairs))
        if abs(observed - metrics["paired_feedback_delta"][metric]) > 1e-12:
            raise ValueError("Independent feedback delta calculation disagrees")
    return {"maximum_method_average_difference": maximum_difference,
            "per_query": per_query, "feedback_per_query_deltas": pairs, "feedback_outcomes": outcomes,
            "non_feedback_ranking_changes": non_feedback_changes,
            "paired_design": "Same query, playlist and hard exclusions; hybrid without versus with supplied soft feedback. This is a controlled feedback ablation, not a longitudinal user study."}


def finalize(directory, human_reviews, output_dir, *, ai_decisions=None, reference_dirs=None):
    directory, output = Path(directory), Path(output_dir)
    if output.exists():
        raise FileExistsError("Use a fresh output directory; prior evidence is immutable")
    run_path, judgment_path = directory / "runs.json", directory / "judgments.csv"
    document, rows = json.loads(run_path.read_text()), read_rows(judgment_path)
    validate_pool(document, rows)
    origins = reference_dirs or {
        "tobi": Path("evaluation/part4/adaptive_evaluation_v1"),
        "janani": Path("evaluation/part4/janani_results"),
    }
    reference_rows, reference_paths = {}, []
    for owner, path in origins.items():
        reference = path / ("reviewed_judgments.csv" if owner == "tobi" else "judgments.csv")
        reference_paths.append(reference)
        for row in read_rows(reference):
            if pair(row) in reference_rows:
                raise ValueError("Ambiguous reference-pool identity")
            reference_rows[pair(row)] = row
    for row in rows:
        if pair(row) not in reference_rows:
            reference_rows[pair(row)] = row
    reviews, imports, repairs, comparison_pairs = {}, [], [], []
    for review_path in map(Path, human_reviews):
        submitted = read_rows(review_path)
        pairs, file_repairs = [], []
        for row in submitted:
            key = pair(row)
            if key not in reference_rows:
                raise ValueError(f"Submitted review not found in original frozen pools: {key}")
            fixed, changes = canonical_review(row, reference_rows[key])
            if key in reviews:
                raise ValueError("Overlapping human review files require explicit reconciliation")
            reviews[key] = {**fixed, "review_source": review_path.name,
                            "review_source_sha256": sha256(review_path)}
            file_repairs.extend({"query_id": key[0], "track_id": key[1], **c} for c in changes)
            if reference_rows[key]["grade"] in {"0", "1", "2"}:
                pairs.append((reference_rows[key]["grade"], fixed["grade"]))
        query_counts = Counter(row["query_id"] for row in submitted)
        imports.append({"source": review_path.name, "sha256": sha256(review_path),
                        "assessors": dict(Counter(r["assessor"] for r in submitted)),
                        "reviewed_queries": len(query_counts), "pairs_per_query": dict(sorted(query_counts.items())),
                        "agreement_with_previous_ai": agreement(pairs), "metadata_repairs": len(file_repairs)})
        comparison_pairs.extend(pairs)
        repairs.extend(file_repairs)
    merged, overrides = [], []
    for original in rows:
        row = dict(original)
        row["original_grade"], row["original_assessor"] = row["grade"], row["assessor"]
        if pair(row) in reviews:
            reviewed, changes = canonical_review(reviews[pair(row)], row)
            if changes:
                raise ValueError("Normalized human evidence unexpectedly differs in current pool")
            row.update({field: reviewed[field] for field in ("grade", "reason", "assessor", "review_source", "review_source_sha256")})
            row["assessment_type"] = "submitted_human"
            overrides.append({"query_id": row["query_id"], "track_id": row["track_id"],
                              "previous_ai_grade": original["grade"], "human_grade": row["grade"]})
        else:
            row["assessment_type"] = "unjudged" if not row["grade"] else "reused_ai_metadata"
        merged.append(row)
    if ai_decisions:
        decisions = json.loads(Path(ai_decisions).read_text())
        if decisions["runs_sha256"] != sha256(run_path) or decisions["pool_sha256"] != sha256(judgment_path):
            raise ValueError("AI decision manifest does not match frozen inputs")
        if not decisions["assessor"].casefold().startswith(("codex-ai:", "ai-assisted:", "ai:")):
            raise ValueError("Draft decisions require an explicit AI assessor")
        supplied = {(d["query_id"], d["track_id"]): d for d in decisions["decisions"]}
        missing = {pair(row) for row in merged if not row["grade"]}
        if len(supplied) != len(decisions["decisions"]) or supplied.keys() != missing:
            raise ValueError("Explicit AI decisions must cover exactly the unjudged pairs")
        for row in merged:
            if pair(row) not in supplied:
                continue
            decision = supplied[pair(row)]
            if type(decision["grade"]) is not int or decision["grade"] not in (0, 1, 2) or not decision["reason"].strip():
                raise ValueError("Every AI draft needs an explicit integer 0–2 grade and reason")
            row.update(grade=str(decision["grade"]), reason=decision["reason"],
                       assessor=decisions["assessor"], assessment_type="new_ai_metadata")
    pending = [row for row in merged if not row["grade"]]
    report = {
        "evaluation": "Integrated standard application route, common four-method pool",
        "runs_sha256": sha256(run_path), "pool_sha256": sha256(judgment_path),
        "query_count": len(document["queries"]), "ranking_count": len(document["runs"]),
        "timing_samples": sum(len(run["latency_ms"]) for run in document["runs"]),
        "pooled_pairs": len(merged), "human_review_imports": imports,
        "submitted_review_count": len(reviews), "combined_ai_human_agreement": agreement(comparison_pairs),
        "metadata_repairs": repairs, "human_overrides": overrides,
        "assessment_counts": dict(Counter(r["assessment_type"] for r in merged)),
        "missing_pairs": len(pending), "fully_human_judged_pool": all(r["assessment_type"] == "submitted_human" for r in merged),
        "review_process_note": "Submitted reviewer identities are retained. Independence, listening and blinding were not independently verified. Sample agreement is not full-pool validation or inter-human agreement.",
        "query_categories": dict(Counter(q["category"] for q in document["queries"])),
        "excluded_tracks_returned": sum(r["final_excluded_count"] for r in document["runs"]),
        "latency": {m: latency_summary([t for r in document["runs"] if r["method"] == m for t in r["latency_ms"]]) for m in METHODS},
        "latency_scope": document["latency_scope"],
        "metrics": None if pending else score_rows(run_path, merged),
        "source_hashes": {str(p): sha256(p) for p in [run_path, judgment_path, *reference_paths, *map(Path, human_reviews)]},
        "evaluator_source_sha256": {str(p): sha256(p) for p in map(Path, (
            "retrieval/evaluation/r4_finalize.py", "retrieval/evaluation/part4.py", "retrieval/evaluation/metrics.py"))},
    }
    protocol_path = Path("evaluation/part4/human_review/sampling_protocol.json")
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        planned = {(s["query_id"], t) for s in protocol["strata"] for t in s["selected_track_ids"]}
        returned = {key for key in reviews if key[0].startswith("tobi-")}
        report["tobi_sample_protocol_check"] = {
            "planned_sample_sha256": sha256(protocol_path), "planned_pairs": len(planned),
            "returned_pairs": len(returned), "matching_planned_pairs": len(planned & returned),
            "same_sample": planned == returned,
            "interpretation": "Returned reviews still cover three per query. Do not describe them as the original seeded random sample unless the identities match."}
    report["historical_review_effect"] = {}
    for owner, path in origins.items():
        historical = read_rows(reference_paths[0 if owner == "tobi" else 1])
        corrected = deepcopy(historical)
        for row in corrected:
            if pair(row) in reviews:
                row.update({f: reviews[pair(row)][f] for f in ("grade", "reason", "assessor")})
        report["historical_review_effect"][owner] = {
            "runs_sha256": sha256(path / "runs.json"),
            "before_human_review": score_rows(path / "runs.json", historical),
            "after_human_review": score_rows(path / "runs.json", corrected),
            "scope": "Historical frozen pipeline only. Do not combine these separate pipelines as a unified 50-query experiment."}
    if report["metrics"]:
        report["independent_metric_audit"] = independent_scores(document, merged, report["metrics"])
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source_reviews"
    source_dir.mkdir()
    for i, source in enumerate(map(Path, human_reviews), 1):
        shutil.copy2(source, source_dir / f"{i}_{source.name}")
    if ai_decisions:
        shutil.copy2(ai_decisions, output / "ai_decisions.json")
        report["ai_decisions_sha256"] = sha256(ai_decisions)
    for name, data in (("report.json", report), ("merged_judgments.json", merged),
                       ("remaining_unjudged.json", pending)):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (output / "REPORT.md").write_text(render_report(report))
    return report


def render_report(r):
    lines = ["# R4 comparative evaluation", "",
             f"The frozen integrated run contains {r['query_count']} queries, {r['ranking_count']} rankings and {r['pooled_pairs']} pooled query/track pairs.", "",
             "## Relevance judgments", "",
             f"Imported {r['submitted_review_count']} submitted human reviews. {len(r['human_overrides'])} apply to the current pool; the rest refer to earlier candidate pools.", "",
             f"Human-reviewed coverage of the current pool: {len(r['human_overrides']) / r['pooled_pairs']:.1%}. The human grades change {sum(d['previous_ai_grade'] != d['human_grade'] for d in r['human_overrides'])} of those current-pool assessments.", "",
             f"Current assessment counts: {json.dumps(r['assessment_counts'], sort_keys=True)}.", "",
             "| Review file | Pairs | Queries | Exact agreement with previous AI grades |",
             "| --- | ---: | ---: | ---: |"]
    for source in r["human_review_imports"]:
        a = source["agreement_with_previous_ai"]
        display = "n.a." if a["exact_agreement"] is None else f"{a['exact_agreement']:.1%}"
        count = sum(source["pairs_per_query"].values())
        lines.append(f"| {source['source']} | {count} | {source['reviewed_queries']} | {display} ({a['pairs']} comparable) |")
    lines += ["", r["review_process_note"], "",
              "Human reviews override AI grades only for identical query, track and context evidence. They are never generalized to unreviewed songs. Remaining AI grades are provisional; the human sample is not treated as full-pool ground truth.", "",
              f"Recorded reversible metadata repairs: {len(r['metadata_repairs'])}. Original files and human grades are preserved.", ""]
    sampling = r.get("tobi_sample_protocol_check")
    if sampling and not sampling["same_sample"]:
        lines += [f"Only {sampling['matching_planned_pairs']} of {sampling['returned_pairs']} returned Tobi-pool reviews match the originally exported seeded sample. Each query still has three reviews, but the returned sample must not be described as that seeded random selection. Agreement is descriptive for these submitted pairs.", ""]
    lines += ["## Comparative results", ""]
    if r["metrics"]:
        metrics = r["metrics"]
        lines += [f"**{metrics['judgment_status']}**. This is not a fully human-judged evaluation.", "",
                  "| Method | nDCG@5 | Precision@5 | Mean warm ms | p50 ms | p95 ms | Exclusions returned |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in METHODS:
            m = metrics["methods"][method]
            lines.append(f"| {method} | {m['ndcg@5']:.4f} | {m['precision@5']:.4f} | {m['mean_ms']:.1f} | {m['p50_ms']:.1f} | {m['p95_ms']:.1f} | {m['excluded_tracks_returned']} |")
        delta = metrics["paired_feedback_delta"]
        lines += ["", f"On the {metrics['feedback_query_count']} feedback queries, mean paired hybrid-with-feedback minus hybrid differences are {delta['ndcg@5']:+.4f} nDCG@5 and {delta['precision@5']:+.4f} Precision@5. These are observed differences under mixed judgments, without a significance claim.", ""]
        audit = r["independent_metric_audit"]
        lines += [audit["paired_design"], "",
                  f"Feedback outcomes by query: {json.dumps(audit['feedback_outcomes'], sort_keys=True)}. The independent standard-library calculation agrees with method averages and paired feedback deltas (maximum method-average difference {audit['maximum_method_average_difference']:.2g}).", "",
                  "| Category | BM25 nDCG@5 | Semantic nDCG@5 | Hybrid nDCG@5 | Hybrid + feedback nDCG@5 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for category, methods in metrics["by_category"].items():
            lines.append(f"| {category} | " + " | ".join(f"{methods[m]['ndcg@5']:.4f}" for m in METHODS) + " |")
        lines += [""]
        lines += [f"Queries without feedback whose hybrid ranking changed: {len(audit['non_feedback_ranking_changes'])}. Overall BM25 and hybrid scores are very close; these results do not establish that hybrid without feedback beats BM25. The small difference between hybrid mean latencies does not establish a speed improvement.", ""]
    else:
        lines += [f"nDCG@5 and Precision@5 remain unavailable: {r['missing_pairs']} pooled pairs are unjudged. They are not replaced by zero. The full scoring command remains gated on explicit grades.", "",
                  "| Method | Mean warm ms | p50 ms | p95 ms |",
                  "| --- | ---: | ---: | ---: |"]
        for method, m in r["latency"].items():
            lines.append(f"| {method} | {m['mean_ms']:.1f} | {m['p50_ms']:.1f} | {m['p95_ms']:.1f} |")
        lines += [""]
    lines += [f"Returned excluded tracks: {r['excluded_tracks_returned']}. Timing scope: {r['latency_scope']}.", "",
              f"nDCG@5 uses gains 2^grade − 1 and log2(rank + 1) discounts against the same complete query pool. Precision@5 counts grades 1 and 2 and divides by five, including short lists. Results are macro-averaged over queries. This run contains {r['timing_samples']} timing samples.", "",
              "## Protocol and limits", "",
              "Grades follow the existing rubric: 0 = unsupported/unrelated or a hard-constraint violation; 1 = partial match; 2 = strong support for the full request. Metadata-only judgments are not listening tests. Genre, mood and seed requirements apply together; retrieval scores and system identity are not relevance evidence.", "",
              "A zero due to missing metadata is not proof that the song sounds irrelevant. Folksonomic tags can be sparse, contradictory or incidental. Metadata-based assessment may favor lexical retrieval, and these results do not establish general musical quality.", "",
              f"Query coverage: {json.dumps(r['query_categories'], sort_keys=True)}. The combined set covers all proposal categories, although the partners' category distributions differ.", "",
              "The actual semantic encoder is Qwen/Qwen3-Embedding-0.6B, rather than the proposal's Sentence-BERT. This difference must be disclosed. The benchmark evaluates the standard application route; the separate live Ollama smoke check is integration evidence, not an additional retrieval-quality evaluation.", "",
              "The approved proposal called for manual judgments of pooled candidates. A human-reviewed sample with remaining AI judgments is a methodological deviation. Report it explicitly and obtain course acceptance or finish full manual assessment before presenting this as the proposed fully manual study.", "",
              "No system was retuned using these human reviews. Historical review effects are recorded separately in report.json and are not pooled across different implementations. Source hashes, review imports, all grade overrides and normalized encoding repairs are recorded there.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="evaluation/part4/integrated_evaluation_v2")
    parser.add_argument("--human-reviews", nargs="+", required=True)
    parser.add_argument("--ai-decisions")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = finalize(args.directory, args.human_reviews, args.output_dir, ai_decisions=args.ai_decisions)
    print(json.dumps({key: report[key] for key in ("query_count", "pooled_pairs", "submitted_review_count",
                                                  "assessment_counts", "missing_pairs")}, indent=2))


if __name__ == "__main__":
    main()
