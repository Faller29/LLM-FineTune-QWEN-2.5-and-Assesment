"""Pipeline tests with synthetic responses; these are not model evaluation data."""
from __future__ import annotations
import argparse
import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import ew_core
import evaluate_ew as runner

SOURCE = Path(__file__).resolve().parent


class FakeBackend:
    calls = []
    fail_at = None

    def __init__(self, args):
        self.args = args

    def describe(self):
        return {"backend": "SOFTWARE_TEST_ONLY", "base_model": "SYNTHETIC_TEST_MODEL",
                "adapter_fingerprints": {"weights": "FAKE"}, "model_context_limit": 32768}

    def input_length(self, prompt):
        return 80

    def generate_task(self, prompt, task):
        if self.fail_at is not None and len(self.calls) >= self.fail_at:
            raise RuntimeError("Synthetic interruption")
        self.calls.append((prompt, task["id"]))
        return {"response": "SOFTWARE TEST ONLY. Evidence placeholder. Second error placeholder.",
                "input_tokens": 80, "output_tokens": 12, "finish_reason": "eos", "model_input": prompt}


def fill_review(out, bank, records, name="Synthetic test reviewer", points="maximum"):
    review = ew_core.review_template(out, bank, records)
    review["reviewer_name"] = name
    maxima = {u["id"]: u["maximum_points"] for u in bank["units"]}
    for case in review["cases"]:
        for rating in case["ratings"]:
            rating.update(status="reviewed", base_points=maxima[rating["unit_id"]] if points == "maximum" else 0,
                          evidence="Evidence placeholder." if points == "maximum" else "", reason="Synthetic rubric test.")
            if "closing_components" in rating:
                for part in rating["closing_components"].values():
                    part.update(credited=points == "maximum", evidence="Evidence placeholder." if points == "maximum" else "",
                                reason="Synthetic component test.")
        for group in case["paragraph_errors"]:
            group.update(status="reviewed", scope_qualification_present=True, comment="Synthetic error review.")
    return review


class EWTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = SOURCE / "materials.json"
        if not path.is_file():
            raise unittest.SkipTest("Run prepare_ew.py before these software tests.")
        cls.bank = ew_core.read_json(path)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.here = self.root / "EW_Evaluation"
        self.here.mkdir()
        for name in ["ew_core.py", "evaluate_ew.py", "ew_backend.py", "prepare_ew.py",
                     "ew_manifest.json", "REVIEW_GUIDE.txt", "METHOD_NOTES.txt"]:
            shutil.copyfile(SOURCE / name, self.here / name)
        tsa = self.root / "TSA_Evaluation"
        tsa.mkdir()
        for name in ["evaluate_tsa.py", "tsa_core.py"]:
            shutil.copyfile(SOURCE.parent / "TSA_Evaluation" / name, tsa / name)
        ew_core.write_json(self.here / "materials.json", self.bank)
        self.core_patch = patch.object(ew_core, "HERE", self.here)
        self.runner_patch = patch.object(runner, "HERE", self.here)
        self.core_patch.start()
        self.runner_patch.start()
        FakeBackend.calls, FakeBackend.fail_at = [], None
        self.out = self.root / "output"
        self.args = argparse.Namespace(adapter="fake", model=None, mode="on", max_input=2048,
                  max_new_tokens=2048, threads=4, seed=42, temperatures="0", rounds=1,
                  scope="full", tsa_out=None, out=str(self.out), resume=False,
                  check_only=False, allow_model_download=False)

    def tearDown(self):
        self.core_patch.stop()
        self.runner_patch.stop()
        self.tmp.cleanup()

    def generate(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.run(self.args, FakeBackend)

    def saved(self):
        return ew_core.read_json(self.out / "run_config.json"), ew_core.read_jsonl(self.out / "responses.jsonl")

    def test_full_materials_and_multi_flaw_mapping(self):
        ew_core.validate_bank(self.bank)
        counts = [len([u for u in self.bank["units"] if u["scenario_id"] == s]) for s in ew_core.SCENARIO_IDS]
        self.assertEqual(counts, [9, 5, 4, 3])
        self.assertEqual(sum(u["maximum_points"] for u in self.bank["units"]), 65)

    def test_inputs_exclude_reference_answers(self):
        self.assertEqual(self.generate(), 0)
        for prompt, _ in FakeBackend.calls:
            self.assertIn(prompt, [x["prompt"] for x in self.bank["scenarios"]])
            for unit in self.bank["units"]:
                self.assertNotIn(unit["reference"], prompt)

    def test_generation_finished_but_score_is_pending(self):
        self.generate()
        summary = ew_core.read_json(self.out / "summary.json")
        self.assertTrue(summary["completion"]["complete"])
        self.assertEqual(summary["scoring_status"], "pending")
        self.assertIsNone(summary["scores"]["by_temperature"][0]["mean_score"])
        self.assertTrue((self.out / "Compile_ANSWER.zip").is_file())

    def test_resume_retains_completed_responses(self):
        FakeBackend.fail_at = 1
        self.assertEqual(self.generate(), 1)
        first = (self.out / "responses.jsonl").read_bytes()
        FakeBackend.fail_at = None
        self.args.resume = True
        self.assertEqual(self.generate(), 0)
        self.assertTrue((self.out / "responses.jsonl").read_bytes().startswith(first))
        self.assertEqual(len(FakeBackend.calls), 4)

    def test_resume_changed_budget_rejected(self):
        self.generate()
        self.args.resume = True
        self.args.max_new_tokens += 1
        with self.assertRaisesRegex(ValueError, "Resume"):
            self.generate()

    def test_context_overflow_rejected_without_generation(self):
        self.args.max_input = 64
        with self.assertRaisesRegex(ValueError, "input exceeds"):
            self.generate()
        self.assertEqual(FakeBackend.calls, [])

    def test_review_bound_to_actual_responses(self):
        self.generate()
        _, records = self.saved()
        review = fill_review(self.out, self.bank, records)
        review["responses_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "different/older"):
            ew_core.score_review(self.out, self.bank, records, review)

    def test_review_rejects_fabricated_evidence(self):
        self.generate()
        _, records = self.saved()
        review = fill_review(self.out, self.bank, records)
        review["cases"][0]["ratings"][0]["evidence"] = "NOT IN THE MODEL RESPONSE"
        with self.assertRaisesRegex(ValueError, "exact excerpt"):
            ew_core.score_review(self.out, self.bank, records, review)

    def test_two_errors_charged_once_by_manual(self):
        self.generate()
        config, records = self.saved()
        review = fill_review(self.out, self.bank, records)
        new_case = review["cases"][1]
        new_case["paragraph_errors"][0]["errors"] = [
            {"kind": "faulty_reasoning", "evidence": "Evidence placeholder.", "explanation": "First synthetic error."},
            {"kind": "unsupported_claim", "evidence": "Second error placeholder.", "explanation": "Second synthetic error."}]
        score = ew_core.score_review(self.out, self.bank, records, review)
        totals = ew_core.summarize_scores(config, [score])["mean_across_reviewers_by_round"][0]
        self.assertEqual(totals["mean_manual_score"], 64)
        self.assertEqual(totals["mean_article_wording_sensitivity_score"], 63)

    def test_original_closing_weights_and_scope_omissions(self):
        self.generate()
        config, records = self.saved()
        review = fill_review(self.out, self.bank, records)
        for group in review["cases"][0]["paragraph_errors"]:
            if group["paragraph"] in {3, 8}:
                group["scope_qualification_present"] = False
        score = ew_core.score_review(self.out, self.bank, records, review)
        total = ew_core.summarize_scores(config, [score])["mean_across_reviewers_by_round"][0]
        self.assertEqual(total["mean_manual_score"], 63)
        review["cases"][0]["ratings"][-1]["base_points"] = 4
        with self.assertRaisesRegex(ValueError, "1\\+1\\+2\\+1"):
            ew_core.score_review(self.out, self.bank, records, review)

    def test_negative_deduction_not_silently_clamped(self):
        self.generate()
        config, records = self.saved()
        review = fill_review(self.out, self.bank, records, points="zero")
        review["cases"][0]["paragraph_errors"][0]["errors"] = [
            {"kind": "faulty_reasoning", "evidence": "Evidence placeholder.", "explanation": "Synthetic error."}]
        total = ew_core.summarize_scores(config, [ew_core.score_review(self.out, self.bank, records, review)])
        self.assertEqual(total["mean_across_reviewers_by_round"][0]["mean_manual_score"], -1)

    def test_review_cannot_drop_a_flaw(self):
        self.generate()
        _, records = self.saved()
        review = fill_review(self.out, self.bank, records)
        review["cases"][1]["ratings"].pop()
        with self.assertRaisesRegex(ValueError, "every scoring unit"):
            ew_core.score_review(self.out, self.bank, records, review)

    def test_temperature_schedule_and_seeds(self):
        plan = ew_core.make_plan(self.bank, [0, 0.25, 0.5, 0.75, 1], 30, 42)
        self.assertEqual(len(plan), 600)
        self.assertEqual(len({x["id"] for x in plan}), 600)
        self.assertEqual(len({x["generation_seed"] for x in plan}), 600)

    def test_summary_export_after_import(self):
        self.generate()
        config, records = self.saved()
        review = fill_review(self.out, self.bank, records)
        ew_core.write_json(self.out / "accepted_review_01.json", review)
        summary = runner.export(self.out, self.bank, config, records)
        self.assertEqual(summary["scoring_status"], "reviewed")
        self.assertEqual(summary["scores"]["by_temperature"][0]["mean_score"], 65)
        self.assertTrue((self.out / "scenario_scores.csv").is_file())

    def test_stale_review_excluded_after_resume(self):
        FakeBackend.fail_at = 1
        self.generate()
        _, records = self.saved()
        ew_core.write_json(self.out / "accepted_review_01.json", fill_review(self.out, self.bank, records))
        FakeBackend.fail_at = None
        self.args.resume = True
        self.assertEqual(self.generate(), 0)
        summary = ew_core.read_json(self.out / "summary.json")
        self.assertEqual(summary["scoring_status"], "pending")
        self.assertEqual(summary["outdated_reviews_excluded"], ["accepted_review_01.json"])


if __name__ == "__main__":
    unittest.main()
