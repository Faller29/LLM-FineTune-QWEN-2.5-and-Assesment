"""Software tests only; mock responses are never Qwen performance evidence."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import evaluate_tsa as runner
import prepare_tsa
from tsa_core import (
    CRITERIA, binomial_upper, digest, option_block, parse_answer, prompt_for,
    read_jsonl, score, validate_bank, validate_records,
)


def fixture_bank():
    rows = []
    for i in range(35):
        text = (f"Software test fixture {i + 1}, not an assessment question.\n"
                "A. alpha\nB. beta\nC. gamma\nD. delta\nE. epsilon")
        rows.append({"id": f"FIXTURE-{i:02}", "criterion": CRITERIA[i // 5],
                     "gold": "ABCDE"[i % 5], "text": text,
                     "question_sha256": digest(text.encode("utf-8")),
                     "year": None, "question_number": None})
    return rows


def response(item, raw="Answer: A\nReason: Software test fixture."):
    return {"id": item["id"], "prompt": prompt_for(item), "response": raw,
            "parsed": parse_answer(raw), "finish_reason": "eos", "seconds": 0.1}


class FakeBackend:
    def __init__(self, args):
        self.calls = 0

    def describe(self):
        return {"base_model": "SOFTWARE_FIXTURE_NOT_QWEN"}

    def input_length(self, prompt):
        return 100

    def generate(self, prompt):
        self.calls += 1
        return {"response": "Answer: A\nReason: This is a software fixture.",
                "model_input": prompt, "input_tokens": 100, "output_tokens": 12,
                "finish_reason": "eos"}


class TestAnswers(unittest.TestCase):
    def test_valid_formats_and_closed_trace(self):
        for raw in ["Answer: C", "**Answer: C.**\nReason: text", "(C)\nReason: text",
                    "Answer: C because the premise supports it.",
                    "The correct answer is C.", "<think>Answer: A</think>\nAnswer: C"]:
            self.assertEqual(parse_answer(raw)["choice"], "C", raw)

    def test_no_guesses_from_options_or_reason(self):
        for raw in ["I discussed C but did not select an answer.",
                    "A. alpha\nB. beta\nC. gamma\nD. delta\nE. epsilon",
                    "Answer: A, B, C, D, or E",
                    "UNKNOWN", "", "Answer: A or B", "Answer: A / B"]:
            self.assertIsNone(parse_answer(raw)["choice"], raw)

    def test_conflicting_answers_and_unfinished_trace(self):
        self.assertEqual(parse_answer("Answer: A\nAnswer: B")["status"],
                         "conflicting_answer_markers")
        self.assertIsNone(parse_answer("<thinking>Answer: C")["choice"])

    def test_options_do_not_use_earlier_A_sentence(self):
        text = "A scientist made a claim.\nQuestion?\nA. aa\nB. bb\nC. cc\nD. dd\nE. ee"
        self.assertEqual(option_block(text)["A"], "aa")


class TestScores(unittest.TestCase):
    def setUp(self):
        self.items = fixture_bank()

    def test_bank_and_tamper_rejection(self):
        validate_bank(self.items)
        self.items[0]["text"] += " changed"
        with self.assertRaises(ValueError):
            validate_bank(self.items)

    def test_constant_letter_does_not_look_like_broad_success(self):
        result = score(self.items, [response(x) for x in self.items])
        self.assertEqual(result["answer_key_score"]["earned_points_so_far"], 7)
        self.assertEqual(result["answer_key_score"]["accuracy"], 0.2)
        self.assertTrue(all(x["correct"] == 1 for x in result["criteria"]))

    def test_all_invalid_and_budget_flags(self):
        records = [response(x, "UNKNOWN") for x in self.items]
        records[0]["finish_reason"] = "length"
        result = score(self.items, records)
        self.assertEqual(result["answer_key_score"]["invalid"], 35)
        self.assertEqual(result["answer_key_score"]["earned_points_so_far"], 0)
        self.assertEqual(result["output_budget_hits"], 1)
        self.assertEqual(result["explanation_review"]["status"], "not_scored")

    def test_partial_score_not_final(self):
        result = score(self.items, [response(self.items[0])])
        self.assertFalse(result["completion"]["complete"])
        self.assertIsNone(result["answer_key_score"]["accuracy"])
        self.assertEqual(len(result["completion"]["missing_ids"]), 34)

    def test_records_reject_duplication_and_changed_prompts(self):
        row = response(self.items[0])
        with self.assertRaises(ValueError):
            validate_records(self.items, [row, row])
        row["prompt"] += " changed"
        with self.assertRaises(ValueError):
            validate_records(self.items, [row])

    def test_binomial_boundary(self):
        self.assertAlmostEqual(binomial_upper(0, 35), 1)
        self.assertAlmostEqual(binomial_upper(35, 35), 0.2 ** 35)


class TestImport(unittest.TestCase):
    def test_inline_and_shared_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.xlsx"
            with ZipFile(path, "w") as z:
                z.writestr("xl/workbook.xml",
                           '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                           '<sheets><sheet name="TSA_original" r:id="rId1"/></sheets></workbook>')
                z.writestr("xl/_rels/workbook.xml.rels",
                           '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
                           '</Relationships>')
                z.writestr("xl/sharedStrings.xml",
                           '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                           '<si><r><t>Shared </t></r><r><t>text</t></r></si></sst>')
                z.writestr("xl/worksheets/sheet1.xml",
                           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                           '<sheetData><row r="2"><c r="A2" t="s"><v>0</v></c>'
                           '<c r="B2" t="inlineStr"><is><t>Inline text</t></is></c></row>'
                           '</sheetData></worksheet>')
            self.assertEqual(prepare_tsa.read_sheet(path, "TSA_original")[2],
                             {"A": "Shared text", "B": "Inline text"})


class TestWorkflow(unittest.TestCase):
    def args(self, out, resume=False):
        return argparse.Namespace(
            smoke=False, out=str(out), resume=resume, adapter="unused_test_path",
            model=None, mode="on", max_input=2048, max_new_tokens=512,
            threads=4, seed=42, allow_model_download=False,
        )

    def test_end_to_end_resume_and_review(self):
        bank = fixture_bank()
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "load_items", return_value=bank):
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(self.args(out), FakeBackend), 0)
                self.assertEqual(runner.run(self.args(out, True), FakeBackend), 0)
            self.assertEqual(len(read_jsonl(out / "responses.jsonl")), 35)
            self.assertTrue((out / "Compile_Answer.zip").is_file())
            with ZipFile(out / "Compile_Answer.zip") as z:
                self.assertIn("review_packet.txt", z.namelist())
            changed = self.args(out, True)
            changed.mode = "off"
            with self.assertRaises(ValueError):
                runner.run(changed, FakeBackend)
            ratings = json.loads((out / "review_template.json").read_text(encoding="utf-8"))
            ratings["reviewer_name"] = "Software fixture"
            for row in ratings["ratings"]:
                row.update(answer_status="clear", choice_as_written="A",
                           answer_evidence="Answer: A", explanation_status="insufficient",
                           explanation_evidence="This is a software fixture.",
                           comment="Fixture does not justify a real question.")
            rp = out / "completed.json"
            rp.write_text(json.dumps(ratings), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                runner.apply_review(out, rp)
            reviewed = json.loads((out / "summary_reviewed.json").read_text(encoding="utf-8"))
            self.assertEqual(reviewed["review"]["audited_answer_points"], 7)
            self.assertEqual(reviewed["answer_key_score"]["earned_points_so_far"], 7)
            self.assertEqual(reviewed["review"]["explanation_counts"]["insufficient"], 35)
            ratings["ratings"][0]["answer_evidence"] = "fabricated quote"
            rp.write_text(json.dumps(ratings), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.apply_review(out, rp)

    def test_failure_then_resume_without_repeating_saved_answers(self):
        class FailAfterTwo(FakeBackend):
            def generate(self, prompt):
                if self.calls == 2:
                    raise RuntimeError("Planned software-test failure")
                return super().generate(prompt)

        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "load_items",
                                                               return_value=fixture_bank()):
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.run(self.args(out), FailAfterTwo), 1)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIsNone(summary["answer_key_score"]["accuracy"])
            self.assertEqual(summary["completion"]["observed"], 2)
            saved = read_jsonl(out / "responses.jsonl")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(self.args(out, True), FakeBackend), 0)
            self.assertEqual(read_jsonl(out / "responses.jsonl")[:2], saved)
            self.assertTrue((out / "errors.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
