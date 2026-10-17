"""Run the tuned model on complete EW scenarios; export or apply rubric reviews."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from ew_core import (HERE, SCENARIO_IDS, digest, signature, read_json, read_jsonl,
                     write_json, validate_bank, make_plan, validate_run, review_template,
                     score_review, summarize_scores)


def fingerprints():
    files = [HERE / name for name in ["ew_core.py", "prepare_ew.py", "ew_backend.py",
                                      "evaluate_ew.py", "ew_manifest.json"]]
    tsa = HERE.parent / "TSA_Evaluation"
    files += [tsa / "evaluate_tsa.py", tsa / "tsa_core.py"]
    return {str(p.relative_to(HERE.parent)): digest(p.read_bytes()) for p in files}


def tsa_context(path, identity, mode):
    if not path:
        return None
    from ew_backend import existing_tsa_modules
    _, core = existing_tsa_modules()
    root = Path(path).expanduser().resolve()
    config = read_json(root / "run_config.json")
    bank = read_json(root / "questions_and_keys.json")
    rows = core.read_jsonl(root / "responses.jsonl")
    result = core.score(bank, rows)
    if not result["completion"]["complete"] or len(bank) != 35:
        raise ValueError("The linked TSA run must be the completed 35-item assessment.")
    if digest(json.dumps(bank, sort_keys=True).encode("utf-8")) != config["items_sha256"]:
        raise ValueError("Linked TSA questions changed after its run.")
    other = config["identity"]
    for field in ["base_model", "adapter_fingerprints"]:
        if other.get(field) != identity.get(field):
            raise ValueError(f"Linked TSA used a different model/adapter: {field}")
    if other.get("base_model_revision") and identity.get("base_model_revision"):
        if other["base_model_revision"] != identity["base_model_revision"]:
            raise ValueError("Linked TSA used a different base-model revision.")
    if config["settings"]["mode"] != mode:
        raise ValueError("Linked TSA used a different thinking-mode system instruction.")
    return {"same_model_and_adapter_verified": True, "path": str(root),
            "run_config_sha256": digest((root / "run_config.json").read_bytes()),
            "responses_sha256": digest((root / "responses.jsonl").read_bytes()),
            "automatic_answer_points": result["answer_key_score"]["earned_points_so_far"],
            "maximum_points": 35, "accuracy": result["answer_key_score"]["accuracy"],
            "note": "TSA and EW are reported separately; they are not added into a new composite score."}


def export(out, bank, config, records):
    validate_run(bank, config, records)
    complete = len(records) == len(config["plan"])
    current_response_hash = digest((out / "responses.jsonl").read_bytes())
    current_config_hash = digest((out / "run_config.json").read_bytes())
    review_paths, stale_reviews, reviewed = [], [], []
    for path in sorted(out.glob("accepted_review_*.json")):
        review = read_json(path)
        if review.get("responses_sha256") != current_response_hash or review.get("run_config_sha256") != current_config_hash:
            stale_reviews.append(path.name)
        else:
            reviewed.append(score_review(out, bank, records, review))
            review_paths.append(path)
    if len({x["reviewer_name"] for x in reviewed}) != len(reviewed):
        raise ValueError("Reviewer names must distinguish the independently supplied reviews.")
    statistics_result = summarize_scores(config, reviewed)
    maximum = 65 if config["settings"]["scope"] == "full" else 29
    summary = {"protocol": "Jiang EW materials, English, complete scenario per fresh chat",
               "study_doi": bank["study_doi"],
               "scope": config["settings"]["scope"],
               "scoring_units_per_round": 21 if config["settings"]["scope"] == "full" else 9,
               "analysis_code_sha256": {name: digest((HERE / name).read_bytes()) for name in ["ew_core.py", "evaluate_ew.py"]},
               "completion": {
                   "saved_scenario_responses": len(records), "planned_scenario_responses": len(config["plan"]),
                   "complete": complete}, "nominal_maximum_per_round": maximum,
               "scoring_status": "reviewed" if complete and statistics_result["all_ratings_complete"] else "pending",
               "primary_scoring_policy": "released_manual: at most one deduction per response paragraph",
               "sensitivity_policy": "article wording: one deduction per documented error instance",
               "reviewers": [{"name": r["reviewer_name"], "type": r["reviewer_type"]} for r in reviewed],
               "outdated_reviews_excluded": stale_reviews,
               "scores": statistics_result,
               "output_budget_hits": sum(x["finish_reason"] == "length" for x in records),
               "generation_seconds": sum(x.get("seconds", 0) for x in records),
               "tsa_context": config.get("tsa_context"),
               "limitations": [
                   "This completes both assessment components in a reduced project protocol, not the full experiment.",
                   "TSA link covers 35 original items; the publication used 70 TSA items.",
                   "Default EW run uses one completion per scenario; the publication used 150 rounds per model.",
                   "Source workbook is English; public-item pretraining exposure cannot be ruled out.",
                   "Scoring-manual deduction cap differs from the article's per-error wording; both views are labeled.",
                   "A single AI-assisted reviewer does not reproduce three trained human raters or their reliability.",
                   "No rater-group equating or direct comparison to the publication's calibrated totals is performed.",
                   "No global reasoning/critical-thinking pass threshold or internal-process proof is inferred."]}
    write_json(out / "summary.json", summary)
    write_json(out / "review_template.json", review_template(out, bank, records))
    lines = ["ENNIS-WEIR RESPONSE REVIEW PACKET", "Use the released reference manual; no automatic essay grading.",
             "Two scoring views are reported: released-manual capped deductions and article-wording sensitivity.",
             "An omitted response part must be graded explicitly; pending is not a model error label.", ""]
    byid = {x["id"]: x for x in bank["scenarios"]}
    for row in records:
        lines += ["=" * 65, row["id"], f"Finish: {row['finish_reason']}; output tokens: {row['output_tokens']}",
                  "EXACT MODEL PROMPT", row["prompt"], "RAW MODEL RESPONSE", row["response"],
                  "REFERENCE ANSWERS FOR REVIEW ONLY"]
        for unit in bank["units"]:
            if unit["scenario_id"] == row["scenario_id"]:
                lines += [f"{unit['id']} | maximum {unit['maximum_points']} points", unit["reference"], ""]
    (out / "review_packet.txt").write_text("\n".join(lines), encoding="utf-8")
    report = ["ENNIS-WEIR EVALUATION", f"Responses saved: {len(records)}/{len(config['plan'])}",
              f"Nominal maximum per complete round: {maximum}",
              f"Essay scoring: {summary['scoring_status']}",
              f"Output budget reached: {summary['output_budget_hits']}",
              "Scores remain pending until evidence-based rubric reviews are imported.",
              "No confusion matrix or keyword-based reasoning verdict is generated."]
    for row in statistics_result["mean_across_reviewers_by_round"]:
        if row["complete"]:
            report += [f"T={row['temperature']:g}, round {row['round']}: "
                       f"manual score {row['mean_manual_score']:g}/{maximum}; "
                       f"article-wording sensitivity {row['mean_article_wording_sensitivity_score']:g}/{maximum}"]
    if config.get("tsa_context"):
        ctx = config["tsa_context"]
        report += [f"Linked TSA: {ctx['automatic_answer_points']}/35 "
                   f"({100 * ctx['accuracy']:.2f}%). Same model and adapter verified."]
    report += ["", "Read REVIEW_GUIDE.txt and METHOD_NOTES.txt before interpreting the score."]
    (out / "RESULTS.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not reviewed:
        for name in ["scoring_details.json", "scenario_scores.csv"]:
            (out / name).unlink(missing_ok=True)
    if reviewed and reviewed[0]["cases"]:
        write_json(out / "scoring_details.json", reviewed)
        with (out / "scenario_scores.csv").open("w", newline="", encoding="utf-8") as stream:
            header = ["reviewer_name", "reviewer_type"] + list(reviewed[0]["cases"][0])
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            for reviewer in reviewed:
                for case in reviewer["cases"]:
                    writer.writerow({"reviewer_name": reviewer["reviewer_name"],
                                     "reviewer_type": reviewer["reviewer_type"], **case})
    write_json(out / "COMBINED_ASSESSMENT.json", {"tsa": config.get("tsa_context"),
               "ew": {"scoring_status": summary["scoring_status"], "maximum_per_round": maximum,
                      "scores": statistics_result}, "combined_total": None,
               "interpretation": "Describe measured performance and specific strengths/errors; no binary capability certificate."})
    files = ["run_config.json", "materials_snapshot.json", "responses.jsonl", "summary.json",
             "review_template.json", "review_packet.txt", "RESULTS.txt", "COMBINED_ASSESSMENT.json"]
    files += [p.name for p in review_paths]
    for name in ["errors.jsonl", "scoring_details.json", "scenario_scores.csv"]:
        if (out / name).exists():
            files.append(name)
    with ZipFile(out / "COMPILE_ANSWER.zip", "w", ZIP_DEFLATED) as z:
        for name in files:
            z.write(out / name, name)
        for name in ["REVIEW_GUIDE.txt", "METHOD_NOTES.txt", "ew_manifest.json"]:
            z.write(HERE / name, name)
    return summary


def run(args, factory=None):
    if not 1 <= args.rounds <= 30 or args.threads < 1 or not 64 <= args.max_input <= 32768:
        raise ValueError("Invalid rounds, threads, or input allowance.")
    if not 1 <= args.max_new_tokens <= 4096 or args.seed < 0:
        raise ValueError("Invalid output allowance or seed.")
    temperatures = [float(x) for x in args.temperatures.split(",")]
    if not temperatures or len(set(temperatures)) != len(temperatures) or any(not math.isfinite(t) or not 0 <= t <= 1 for t in temperatures):
        raise ValueError("Use distinct temperatures from 0 to 1, e.g. 0 or 0,0.25,0.5,0.75,1.")
    bank = read_json(HERE / "materials.json")
    validate_bank(bank)
    plan = make_plan(bank, temperatures, args.rounds, args.seed, args.scope)
    if factory is None:
        from ew_backend import backend_factory
        factory = backend_factory
    backend = factory(args)
    lengths = {s["id"]: backend.input_length(s["prompt"]) for s in bank["scenarios"]
               if args.scope == "full" or s["id"] == "EW-ORIGINAL"}
    identity = backend.describe()
    if any(v > args.max_input for v in lengths.values()):
        raise ValueError(f"Complete input exceeds --max-input: {lengths}. No input was truncated.")
    context_limit = identity.get("model_context_limit")
    if context_limit and max(lengths.values()) + args.max_new_tokens > context_limit:
        raise ValueError("Full input plus output allowance exceeds the actual model context limit.")
    config = {"schema_version": 1, "purpose": "EW_assessment", "identity": identity,
              "code_fingerprints": fingerprints(), "materials_sha256": signature(bank),
              "settings": {"scope": args.scope, "temperatures": temperatures, "rounds": args.rounds,
                           "mode": args.mode, "system_prompt": "detailed thinking " + args.mode,
                           "max_input": args.max_input, "max_new_tokens": args.max_new_tokens,
                           "threads": args.threads, "seed": args.seed, "fresh_chat_per_scenario": True,
                           "reference_answers_sent_to_model": False,
                           "sampling_at_positive_temperatures": {"top_p": 1.0, "top_k": 0}},
              "input_tokens": lengths, "plan": plan,
              "tsa_context": tsa_context(args.tsa_out, identity, args.mode)}
    print(f"Preflight passed: {len(plan)} scenario responses planned; longest input {max(lengths.values())} tokens.", flush=True)
    if args.check_only:
        print("Tokenizer, source layout, adapter files, and context allowance checked. No inference performed.")
        return 0
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "responses.jsonl").exists() and not (out / "run_config.json").exists():
        raise ValueError("Existing responses have no configuration. Use a fresh output folder.")
    if args.resume:
        if not (out / "run_config.json").is_file() or read_json(out / "run_config.json") != config:
            raise ValueError("Resume settings/model/source/code differ. Restore them or use a new output folder.")
        if read_json(out / "materials_snapshot.json") != bank:
            raise ValueError("Saved materials differ from the prepared source snapshot.")
    else:
        if (out / "run_config.json").exists():
            raise ValueError("This output already contains a run. Use --resume or a new output folder.")
        write_json(out / "run_config.json", config)
        write_json(out / "materials_snapshot.json", bank)
    records = read_jsonl(out / "responses.jsonl")
    validate_run(bank, config, records)
    scenarios = {x["id"]: x for x in bank["scenarios"]}
    failed, interrupted = None, False
    try:
        with (out / "responses.jsonl").open("a", encoding="utf-8") as stream:
            for task in plan[len(records):]:
                prompt = scenarios[task["scenario_id"]]["prompt"]
                print(f"[{len(records)+1}/{len(plan)}] {task['id']}", flush=True)
                started = time.perf_counter()
                value = backend.generate_task(prompt, task)
                row = {**task, "prompt": prompt, **value,
                       "seconds": round(time.perf_counter() - started, 3),
                       "time_utc": datetime.now(timezone.utc).isoformat()}
                # Check backend output before committing it to the durable response log.
                validate_run(bank, config, records + [row])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                records.append(row)
                print(f"  saved; finish={row['finish_reason']}, output={row['output_tokens']} tokens", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted. Completed scenarios are preserved; resume with identical settings.")
    except Exception as exc:
        failed = exc
        with (out / "errors.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"error": str(exc), "type": type(exc).__name__}) + "\n")
        print(f"Run stopped: {exc}", file=sys.stderr)
    summary = export(out, bank, config, records)
    print(f"Saved {len(records)}/{len(plan)} responses and COMPILE_ANSWER.zip. Essay scoring: {summary['scoring_status']}.")
    return 130 if interrupted else 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--adapter", default="results/qwen_lora/best")
    p.add_argument("--model", default=None)
    p.add_argument("--mode", choices=["on", "off"], default="on")
    p.add_argument("--max-input", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperatures", default="0")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--scope", choices=["full", "original"], default="full")
    p.add_argument("--tsa-out", default=None, help="Optional completed TSA output; verify same adapter and include its score.")
    p.add_argument("--out", default="results/ew_final")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--allow-model-download", action="store_true")
    p = sub.add_parser("summarize")
    p.add_argument("--out", default="results/ew_final")
    p = sub.add_parser("apply-review")
    p.add_argument("--out", default="results/ew_final")
    p.add_argument("--ratings", nargs="+", required=True, help="One or more independently completed review JSON files.")
    args = parser.parse_args()
    try:
        if args.command == "run":
            return run(args)
        out = Path(args.out).expanduser().resolve()
        bank, config = read_json(out / "materials_snapshot.json"), read_json(out / "run_config.json")
        records = read_jsonl(out / "responses.jsonl")
        validate_run(bank, config, records)
        if args.command == "apply-review":
            incoming = [read_json(p) for p in args.ratings]
            for review in incoming:
                score_review(out, bank, records, review)
            if len({x["reviewer_name"] for x in incoming}) != len(incoming):
                raise ValueError("Duplicate reviewer name.")
            for old in out.glob("accepted_review_*.json"):
                old.unlink()
            for number, review in enumerate(incoming, 1):
                write_json(out / f"accepted_review_{number:02}.json", review)
        summary = export(out, bank, config, records)
        print(f"Export rebuilt without model inference. Scoring status: {summary['scoring_status']}.")
        return 0
    except (OSError, ValueError, KeyError, IndexError, ImportError) as exc:
        print(f"Stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Use the existing project environment. Read COMMANDS_WINDOWS_CMD.txt for the correct stage.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
