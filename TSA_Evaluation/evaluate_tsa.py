"""Run one fine-tuned model, save exact responses, and score TSA answer choices."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from tsa_core import (
    CRITERIA, digest, option_block, parse_answer, prompt_for, read_jsonl,
    score, validate_bank, validate_records, write_json,
)

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EXPLANATION_STATUSES = {"pending", "supported", "flawed", "insufficient", "missing", "unclear"}
ANSWER_STATUSES = {"pending", "clear", "ambiguous", "missing"}


def adapter_fingerprints(directory):
    config = directory / "adapter_config.json"
    files = sorted([config] + list(directory.glob("*.safetensors")) + list(directory.glob("*.bin")))
    if not config.is_file() or len(files) < 2:
        raise ValueError("Adapter config or weights missing. Point --adapter at your trained best folder.")
    return {p.name: digest(p.read_bytes()) for p in files}


class DirectBackend:
    """Uses the project's installed Transformers, PEFT and CPU PyTorch."""
    def __init__(self, args):
        import torch
        from transformers import AutoConfig, AutoTokenizer
        self.torch = torch
        self.args = args
        self.adapter = Path(args.adapter).expanduser().resolve()
        self.fingerprints = adapter_fingerprints(self.adapter)
        adapter_config = json.loads((self.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        self.model_name = args.model or adapter_config.get("base_model_name_or_path") or DEFAULT_MODEL
        if args.model and adapter_config.get("base_model_name_or_path") not in (None, args.model):
            raise ValueError("--model differs from the adapter's recorded base model.")
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        local = not args.allow_model_download
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local)
        self.base_config = AutoConfig.from_pretrained(self.model_name, local_files_only=local)
        self.model = None
        # Record the actual packages rather than pretending to run a newer version.
        self.packages = {name: importlib.metadata.version(name)
                         for name in ["torch", "transformers", "peft"]}

    def chat_input(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "detailed thinking " + self.args.mode},
             {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

    def input_length(self, prompt):
        return len(self.tokenizer(self.chat_input(prompt), add_special_tokens=False)["input_ids"])

    def describe(self):
        return {
            "backend": "direct_cpu", "base_model": self.model_name,
            "base_model_revision": getattr(self.base_config, "_commit_hash", None),
            "tokenizer_revision": self.tokenizer.init_kwargs.get("_commit_hash"),
            "adapter_path": str(self.adapter), "adapter_fingerprints": self.fingerprints,
            "packages": self.packages, "dtype": "float32", "do_sample": False,
            "device": "cpu", "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(),
            "model_context_limit": getattr(self.base_config, "max_position_embeddings", None),
        }

    def generate(self, prompt):
        torch = self.torch
        if self.model is None:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
            print("Loading cached base weights and your trained LoRA adapter on CPU...", flush=True)
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name, local_files_only=not self.args.allow_model_download,
                torch_dtype=torch.float32, use_safetensors=True,
            ).to("cpu")
            self.model = PeftModel.from_pretrained(base, str(self.adapter)).eval()
            print("Tuned model loaded. No base-model comparison will be run.", flush=True)
        full = self.chat_input(prompt)
        encoded = self.tokenizer(full, return_tensors="pt", add_special_tokens=False)
        input_tokens = encoded["input_ids"].shape[1]
        if input_tokens > self.args.max_input:
            raise ValueError("Input exceeds the predeclared limit. It was not truncated.")
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded, do_sample=False, max_new_tokens=self.args.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output = generated[0][input_tokens:]
        token_count = int(output.numel())
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
        eos_ids = [v for v in eos_ids if v is not None]
        if token_count and int(output[-1]) in eos_ids:
            finish = "eos"
        elif token_count >= self.args.max_new_tokens:
            finish = "length"
        else:
            finish = "other"
        return {
            "response": self.tokenizer.decode(output, skip_special_tokens=True),
            "input_tokens": input_tokens, "output_tokens": token_count,
            "finish_reason": finish, "model_input": full,
        }


def smoke_item():
    text = (
        "This is only a pipeline check. Select the option whose text is CHECK.\n"
        "A. FIRST\nB. SECOND\nC. CHECK\nD. FOURTH\nE. FIFTH"
    )
    return {"id": "SMOKE-ONLY", "criterion": CRITERIA[0], "gold": "C",
            "text": text, "question_sha256": digest(text.encode("utf-8")),
            "year": None, "question_number": None}


def load_items(smoke=False):
    if smoke:
        return [smoke_item()]
    path = HERE / "questions.json"
    if not path.exists():
        raise ValueError("Questions not prepared. First run python TSA_Evaluation\\prepare_tsa.py")
    items = json.loads(path.read_text(encoding="utf-8"))
    validate_bank(items)
    # Reject edited keys/category assignments as well as edited question text.
    manifest = json.loads((HERE / "item_manifest.json").read_text(encoding="utf-8"))
    reference = {x["id"]: x for x in manifest["items"]}
    for item in items:
        for key, value in reference[item["id"]].items():
            if item.get(key) != value:
                raise ValueError(f"Question metadata/answer key changed: {item['id']}")
    return items


def code_fingerprints():
    return {p.name: digest(p.read_bytes()) for p in
            [HERE / "tsa_core.py", HERE / "evaluate_tsa.py",
             HERE / "prepare_tsa.py", HERE / "item_manifest.json"]}


def make_exports(out, items, records, config):
    if digest(json.dumps(items, sort_keys=True).encode("utf-8")) != config["items_sha256"]:
        raise ValueError("Saved question/key snapshot differs from the original run.")
    if [x["id"] for x in records] != config["planned_order"][:len(records)]:
        raise ValueError("Saved response sequence differs from the recorded question order.")
    summary = score(items, records)
    summary["purpose"] = config["purpose"]
    summary["protocol"] = "TSA original-item subset, English, one completion per item"
    write_json(out / "summary.json", summary)
    template = {
        "reviewer_name": "", "reviewer_type": "Manual",
        "run_config_sha256": digest((out / "run_config.json").read_bytes()),
        "responses_sha256": digest((out / "responses.jsonl").read_bytes()),
        "ratings": [
            {"id": x["id"], "answer_status": "pending", "choice_as_written": None,
             "answer_evidence": "", "explanation_status": "pending",
             "explanation_evidence": "", "comment": ""}
            for x in records
        ],
    }
    write_json(out / "review_template.json", template)
    write_json(out / "questions_and_keys.json", items)
    byid = {x["id"]: x for x in items}
    write_json(out / "item_results.json", [
        {"id": row["id"], "criterion": byid[row["id"]]["criterion"],
         "reference_choice": byid[row["id"]]["gold"],
         "extracted_choice": row["parsed"]["choice"],
         "automatic_points": int(row["parsed"]["choice"] == byid[row["id"]]["gold"]),
         "maximum_points": 1, "extraction_status": row["parsed"]["status"]}
        for row in records
    ])
    lines = [
        "TSA RESPONSE REVIEW PACKET", f"Purpose: {config['purpose']}",
        f"Completed: {len(records)}/{len(items)}",
        "Score choices against the published key. Review explanations separately.",
        "The sources provide answer keys, not a published rationale for every item.",
        "Do not infer a faithful internal process from the written explanation.",
        "Retain the automatic score. Any answer-extraction audit is a separate result.", "",
    ]
    for row in records:
        item = byid[row["id"]]
        options = option_block(item["text"])
        lines += [
            "=" * 65, f"{item['id']} | {item['criterion']}",
            f"Source: TSA {item['year']} question {item['question_number']}",
            "QUESTION", item["text"], "",
            f"REFERENCE ANSWER: {item['gold']}: {options[item['gold']]}",
            f"EXTRACTED ANSWER: {row['parsed']['choice']}",
            f"EXTRACTION STATUS: {row['parsed']['status']}",
            f"AUTOMATIC POINTS: {int(row['parsed']['choice'] == item['gold'])}/1",
            f"FINISH REASON: {row['finish_reason']}", "RAW MODEL RESPONSE",
            row["response"], "",
        ]
    (out / "review_packet.txt").write_text("\n".join(lines), encoding="utf-8")
    report = [
        "TSA answer-key results",
        f"Purpose: {config['purpose']}",
        f"Completed: {len(records)}/{len(items)}",
        f"Points earned so far: {summary['answer_key_score']['earned_points_so_far']}"
        f"/{len(items)}",
        "Final accuracy: " + (
            f"{100 * summary['answer_key_score']['accuracy']:.1f}%"
            if summary["completion"]["complete"] else "unavailable; run incomplete"),
        f"Invalid extracted answers: {summary['answer_key_score']['invalid']}",
        f"Output-token budget reached: {summary['output_budget_hits']}", "",
    ]
    for result in summary["criteria"]:
        report.append(f"{result['criterion']}: {result['correct']}/"
                      f"{result['maximum_points']} ({result['completed']} completed)")
    report += ["", "Explanation review: pending; no additional TSA points.",
               "The raw percentage is not an admissions scaled score.",
               "Use REVIEW_GUIDE.txt when reviewing the explanations."]
    (out / "RESULTS.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    files = ["run_config.json", "responses.jsonl", "summary.json", "RESULTS.txt",
             "questions_and_keys.json", "item_results.json", "review_packet.txt", "review_template.json"]
    if (out / "errors.jsonl").exists():
        files.append("errors.jsonl")
    if (out / "reviewed_ratings.json").exists():
        saved_review = json.loads((out / "reviewed_ratings.json").read_text(encoding="utf-8"))
        if saved_review.get("responses_sha256") == template["responses_sha256"]:
            files += ["reviewed_ratings.json", "summary_reviewed.json"]
    with ZipFile(out / "Compile_Answer.zip", "w", ZIP_DEFLATED) as archive:
        for name in files:
            path = out / name
            if path.exists():
                archive.write(path, name)
        archive.write(HERE / "REVIEW_GUIDE.txt", "REVIEW_GUIDE.txt")
        archive.write(HERE / "item_manifest.json", "item_manifest.json")
    return summary


def run(args, backend_factory=DirectBackend):
    if args.max_input < 64 or args.max_new_tokens < 1 or args.threads < 1:
        raise ValueError("Invalid token or thread limit.")
    if args.max_input > 32768 or args.max_new_tokens > 4096:
        raise ValueError("This CPU protocol limits input to 32768 and output to 4096 tokens.")
    items = load_items(args.smoke)
    shuffled = list(items)
    random.Random(args.seed).shuffle(shuffled)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "run_config.json"
    if (out / "responses.jsonl").exists() and not config_path.exists():
        raise ValueError("Responses without their run configuration: use a fresh output folder.")
    if config_path.exists() and not args.resume:
        raise ValueError("This run already exists. Use --resume or a new --out folder.")
    backend = backend_factory(args)
    lengths = {x["id"]: backend.input_length(prompt_for(x)) for x in items}
    too_long = {k: v for k, v in lengths.items() if v > args.max_input}
    if too_long:
        raise ValueError(f"Full inputs exceed --max-input: {too_long}. No questions were truncated.")
    context_limit = backend.describe().get("model_context_limit")
    if context_limit and max(lengths.values()) + args.max_new_tokens > context_limit:
        raise ValueError("The complete input plus output allowance exceeds the model's context limit.")
    config = {
        "schema_version": 1,
        "purpose": "pipeline_check_only" if args.smoke else "final_35_item_assessment",
        "identity": backend.describe(), "code_fingerprints": code_fingerprints(),
        "items_sha256": digest(json.dumps(items, sort_keys=True).encode("utf-8")),
        "settings": {
            "mode": args.mode, "system_prompt": "detailed thinking " + args.mode,
            "max_input": args.max_input, "max_new_tokens": args.max_new_tokens,
            "threads": args.threads, "seed": args.seed, "do_sample": False,
            "fresh_chat_per_question": True,
        },
        "planned_order": [x["id"] for x in shuffled],
        "input_tokens_by_id": lengths,
    }
    if args.resume:
        if not config_path.exists():
            raise ValueError("No existing run to resume.")
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Model, adapter, code, questions or settings differ from this run. "
                             "Use the original settings or start a separately named run.")
    else:
        write_json(config_path, config)
    records = read_jsonl(out / "responses.jsonl")
    validate_records(items, records)
    completed = {x["id"] for x in records}
    print(f"Preflight passed. Longest complete input: {max(lengths.values())} tokens.")
    print(f"{len(records)}/{len(items)} responses saved. Output folder: {out}", flush=True)
    interrupted, failure = False, None
    try:
        with (out / "responses.jsonl").open("a", encoding="utf-8") as stream:
            for item in shuffled:
                if item["id"] in completed:
                    continue
                prompt = prompt_for(item)
                started = time.perf_counter()
                try:
                    value = backend.generate(prompt)
                except Exception as exc:
                    error = {"id": item["id"], "type": type(exc).__name__,
                             "error": str(exc), "time_utc": datetime.now(timezone.utc).isoformat()}
                    with (out / "errors.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(error, ensure_ascii=False) + "\n")
                    raise
                row = {"id": item["id"], "prompt": prompt, **value,
                       "seconds": round(time.perf_counter() - started, 3),
                       "parsed": parse_answer(value["response"]),
                       "time_utc": datetime.now(timezone.utc).isoformat()}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                records.append(row)
                print(f"[{len(records):02}/{len(items)}] {item['id']} "
                      f"choice={row['parsed']['choice'] or 'INVALID'} "
                      f"finish={row['finish_reason']} {row['seconds']:.1f}s", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saved responses will be preserved.")
    except Exception as exc:
        failure = exc
        print(f"\nRun stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
    summary = make_exports(out, items, records, config)
    print(f"Saved summary and Compile_Answer.zip ({len(records)}/{len(items)} responses).")
    if summary["completion"]["complete"]:
        if args.smoke:
            print("Pipeline check finished; this is not a reasoning/critical-thinking score.")
        else:
            points = summary["answer_key_score"]["earned_points_so_far"]
            print(f"TSA answer-key score: {points}/35 ({100 * points / 35:.1f}%).")
    else:
        print("No final accuracy reported while the run is incomplete. Resume with identical settings.")
    return 130 if interrupted else 1 if failure else 0


def apply_review(out, ratings_path):
    config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    items = json.loads((out / "questions_and_keys.json").read_text(encoding="utf-8"))
    records = read_jsonl(out / "responses.jsonl")
    summary = score(items, records)
    ratings = json.loads(Path(ratings_path).read_text(encoding="utf-8"))
    if ratings.get("run_config_sha256") != digest((out / "run_config.json").read_bytes()):
        raise ValueError("The ratings file belongs to a different run.")
    if ratings.get("responses_sha256") != digest((out / "responses.jsonl").read_bytes()):
        raise ValueError("Responses changed or the run was resumed after this review. Review the current export.")
    if ratings.get("reviewer_type") not in {"human", "Manual"} or not ratings.get("reviewer_name"):
        raise ValueError("Record reviewer_name and reviewer_type: human or Manual.")
    data = ratings.get("ratings", [])
    ids = [x["id"] for x in data]
    if len(set(ids)) != len(ids) or set(ids) != {x["id"] for x in records}:
        raise ValueError("The review must list every saved response exactly once, including pending items.")
    raw = {x["id"]: x["response"] for x in records}
    for row in data:
        if row["answer_status"] not in ANSWER_STATUSES or row["explanation_status"] not in EXPLANATION_STATUSES:
            raise ValueError("Invalid review status.")
        if row["answer_status"] == "clear":
            if row.get("choice_as_written") not in list("ABCDE"):
                raise ValueError("A clear answer requires an A-to-E choice.")
            if not row.get("answer_evidence") or row["answer_evidence"] not in raw[row["id"]]:
                raise ValueError("Answer evidence must be an exact excerpt from the raw response.")
        elif row.get("choice_as_written") is not None:
            raise ValueError("Only a clear answer can have a reviewed choice.")
        evidence = row.get("explanation_evidence", "")
        if evidence and evidence not in raw[row["id"]]:
            raise ValueError("Explanation evidence must be an exact excerpt.")
        if row["explanation_status"] in {"supported", "flawed", "insufficient", "unclear"}:
            if not evidence or not row.get("comment", "").strip():
                raise ValueError("Substantive explanation ratings require evidence and a reason.")
    complete = len(records) == len(items) and all(x["answer_status"] != "pending" for x in data)
    byid = {x["id"]: x for x in items}
    answer_points = sum(x["answer_status"] == "clear"
                        and x["choice_as_written"] == byid[x["id"]]["gold"] for x in data)
    differences = [
        {"id": row["id"], "automatic_choice": next(x["parsed"]["choice"] for x in records if x["id"] == row["id"]),
         "reviewed_choice": row["choice_as_written"]}
        for row in data if row["answer_status"] != "pending"
        and row["choice_as_written"] != next(x["parsed"]["choice"] for x in records if x["id"] == row["id"])
    ]
    summary["review"] = {
        "reviewer_name": ratings["reviewer_name"], "reviewer_type": ratings["reviewer_type"],
        "ratings_sha256": digest(Path(ratings_path).read_bytes()),
        "answer_audit_complete": complete,
        "audited_answer_points": answer_points if complete else None,
        "audited_answer_accuracy": answer_points / len(items) if complete else None,
        "answer_extraction_differences": differences,
        "explanation_counts": dict(Counter(x["explanation_status"] for x in data)),
        "additional_TSA_points": 0,
        "note": "Supplemental reviewer judgments; primary automatic TSA score is preserved. "
                "These explanation categories are project audit labels, not an official TSA scale.",
    }
    write_json(out / "reviewed_ratings.json", ratings)
    write_json(out / "summary_reviewed.json", summary)
    make_exports(out, items, records, config)
    print("Saved summary_reviewed.json. The original automatic score was preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="Generate one answer per question on your local tuned model.")
    p.add_argument("--adapter", default="results/qwen_lora/best")
    p.add_argument("--model", default=None)
    p.add_argument("--mode", choices=["on", "off"], default="on")
    p.add_argument("--max-input", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/tsa_final")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-model-download", action="store_true",
                   help="Optional: allow missing base/tokenizer files to download from Hugging Face.")
    p = sub.add_parser("summarize", help="Rebuild exports from saved responses without running the model.")
    p.add_argument("--out", default="results/tsa_final")
    p = sub.add_parser("apply-review", help="Import separately documented answer/explanation review.")
    p.add_argument("--out", default="results/tsa_final")
    p.add_argument("--ratings", required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            return run(args)
        out = Path(args.out).expanduser().resolve()
        if args.command == "apply-review":
            apply_review(out, args.ratings)
        else:
            config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
            items = json.loads((out / "questions_and_keys.json").read_text(encoding="utf-8"))
            make_exports(out, items, read_jsonl(out / "responses.jsonl"), config)
            print("Rebuilt summaries and Compile_Answer.zip. No model inference was performed.")
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"Stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Use the project's existing .venv. Check the adapter path and cache. "
              "Do not install or upgrade packages merely to change a score.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
