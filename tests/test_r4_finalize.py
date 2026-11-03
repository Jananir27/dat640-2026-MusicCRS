"""Protect submitted reviews, frozen pools, and the distinction from AI drafts."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from retrieval.evaluation.r4_finalize import (
    FIELDS, METHODS, agreement, canonical_review, finalize, read_rows, sha256, validate_pool,
)


def row(query="tobi-1", track="a", grade="2", assessor="AI-assisted:PROVISIONAL"):
    return dict(zip(FIELDS, (query, track, "Recommend rock", "{}", "Ángel", "Maná", "Album", "rock",
                             grade, "Explicit rock metadata", assessor)))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class R4FinalizeTests(unittest.TestCase):
    def test_reversible_encoding_is_logged_but_changed_evidence_and_nonhuman_labels_fail(self):
        source = row()
        review = {**source, "grade": "0", "reason": "Not relevant", "assessor": "Janani"}
        review["title"] = source["title"].encode("utf-8").decode("mac_roman")
        fixed, repairs = canonical_review(review, source)
        self.assertEqual(fixed["title"], "Ángel")
        self.assertEqual(fixed["grade"], "0")
        self.assertEqual(len(repairs), 1)
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            canonical_review({**review, "artist": "Another artist"}, source)
        with self.assertRaisesRegex(ValueError, "cannot be imported"):
            canonical_review({**source, "assessor": "codex-ai:reviewed"}, source)

    def test_agreement_is_ai_to_human_and_zero_is_a_real_grade(self):
        scores = agreement([("2", "2"), ("1", "0"), ("0", "0")])
        self.assertEqual(scores["exact_matches"], 2)
        self.assertAlmostEqual(scores["exact_agreement"], 2 / 3)
        self.assertAlmostEqual(scores["mean_absolute_grade_difference"], 1 / 3)
        self.assertEqual(scores["confusion_ai_rows_human_columns_0_1_2"][1][0], 1)
        self.assertIsNone(agreement([])["exact_agreement"])

    def test_xlsx_reads_literal_zero_and_rejects_formula_grades(self):
        def sheet(formula=False):
            header = "".join(f'<c r="{chr(65+i)}1" t="inlineStr"><is><t>{field}</t></is></c>'
                             for i, field in enumerate(FIELDS))
            values = row(grade="0", assessor="Tobi")
            body = "".join(f'<c r="{chr(65+i)}2" t="inlineStr"><is><t>{values[field]}</t></is></c>'
                           if field != "grade" or not formula else '<c r="I2"><f>1-1</f><v>0</v></c>'
                           for i, field in enumerate(FIELDS))
            return ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                    f'<sheetData><row r="1">{header}</row><row r="2">{body}</row></sheetData></worksheet>')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "review.xlsx"
            with ZipFile(path, "w") as archive:
                archive.writestr("xl/worksheets/sheet1.xml", sheet())
            self.assertEqual(read_rows(path)[0]["grade"], "0")
            with ZipFile(path, "w") as archive:
                archive.writestr("xl/worksheets/sheet1.xml", sheet(formula=True))
            with self.assertRaisesRegex(ValueError, "not formulas"):
                read_rows(path)

    def test_malformed_rankings_or_pools_are_not_reported_as_valid_comparisons(self):
        doc = {"queries": [{"id": "tobi-1"}], "runs": [
            {"query_id": "tobi-1", "method": m, "predicted_track_ids": ["a"], "final_excluded_count": 0}
            for m in METHODS]}
        validate_pool(doc, [row()])
        with self.assertRaisesRegex(ValueError, "unique IDs"):
            validate_pool({**doc, "queries": doc["queries"] * 2}, [row()])
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_pool(doc, [row(), row(track="b")])
        with self.assertRaisesRegex(ValueError, "each common method"):
            validate_pool({**doc, "runs": doc["runs"][:-1]}, [row()])
        doc["runs"][0]["final_excluded_count"] = 1
        with self.assertRaisesRegex(ValueError, "exclusion violation"):
            validate_pool(doc, [row()])

    def test_finalizer_keeps_human_overrides_gates_missing_grades_and_records_ai_drafts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "integrated"
            directory.mkdir()
            origins = {owner: root / owner for owner in ("tobi", "janani")}
            full = [row(), row("janani-1", "b")]
            doc = {"queries": [{"id": q, "category": "genre"} for q in ("tobi-1", "janani-1")],
                   "runs": [{"query_id": q, "category": "genre", "method": m,
                             "predicted_track_ids": [t], "latency_ms": [1.0], "final_excluded_count": 0}
                            for q, t in (("tobi-1", "a"), ("janani-1", "b")) for m in METHODS],
                   "latency_scope": "fixture"}
            (directory / "runs.json").write_text(json.dumps(doc))
            for owner, path in origins.items():
                path.mkdir()
                write_csv(path / ("reviewed_judgments.csv" if owner == "tobi" else "judgments.csv"),
                          [r for r in full if r["query_id"].startswith(owner)])
                (path / "runs.json").write_text(json.dumps({**doc,
                    "queries": [q for q in doc["queries"] if q["id"].startswith(owner)],
                    "runs": [r for r in doc["runs"] if r["query_id"].startswith(owner)]}))
            write_csv(directory / "judgments.csv", [full[0], {**full[1], "grade": "", "assessor": "", "reason": ""}])
            reviews = root / "human.csv"
            write_csv(reviews, [{**full[0], "grade": "0", "assessor": "Janani", "reason": "Not relevant"}])
            result = finalize(directory, [reviews], root / "pending", reference_dirs=origins)
            self.assertIsNone(result["metrics"])
            self.assertEqual(result["missing_pairs"], 1)
            merged = json.loads((root / "pending/merged_judgments.json").read_text())
            self.assertEqual(merged[0]["grade"], "0")
            self.assertEqual(merged[0]["original_grade"], "2")
            manifest = root / "draft.json"
            manifest.write_text(json.dumps({"runs_sha256": sha256(directory / "runs.json"),
                "pool_sha256": sha256(directory / "judgments.csv"), "assessor": "codex-ai:PROVISIONAL",
                "decisions": [{"query_id": "janani-1", "track_id": "b", "grade": 1, "reason": "Partial metadata match"}]}))
            result = finalize(directory, [reviews], root / "scored", ai_decisions=manifest, reference_dirs=origins)
            self.assertEqual(result["metrics"]["query_count"], 2)
            self.assertTrue(result["metrics"]["judgment_status"].startswith("MIXED"))
            self.assertEqual(result["assessment_counts"], {"submitted_human": 1, "new_ai_metadata": 1})
            self.assertFalse(result["fully_human_judged_pool"])
            self.assertTrue((root / "scored/source_reviews/1_human.csv").exists())
            with self.assertRaises(FileExistsError):
                finalize(directory, [reviews], root / "scored", reference_dirs=origins)
            write_csv(reviews, [full[0], full[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_rows(reviews)


if __name__ == "__main__":
    unittest.main()
