"""Small, auditable helpers. No model or third-party imports."""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

CRITERIA = [
    "Main Conclusion", "Drawing a Conclusion", "Identifying an Assumption",
    "Assessing the Impact of Additional Evidence", "Detecting Reasoning Errors",
    "Matching Arguments", "Applying Principles",
]
INSTRUCTION = (
    "Read the question carefully and select the correct option. "
    "Begin your response with Answer: followed by one letter A, B, C, D, or E. "
    "Then give a short reason based on the passage.\n\n"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}: invalid JSON on line {number}; preserve and inspect the file.") from exc
    return rows


def prompt_for(item):
    return INSTRUCTION + item["text"]


def option_block(text):
    """Use the final contiguous A..E block; an earlier paragraph may begin 'A'."""
    matches = list(re.finditer(r"(?m)^[ \t]*([A-E])(?:[.)][ \t]*|[ \t]+)", text))
    starts = [i for i in range(len(matches))
              if [x.group(1) for x in matches[i:i + 5]] == list("ABCDE")]
    if not starts:
        raise ValueError("No complete A-to-E option block.")
    selected = matches[starts[-1]:starts[-1] + 5]
    options = {}
    for j, m in enumerate(selected):
        end = selected[j + 1].start() if j < 4 else len(text)
        value = text[m.end():end].strip()
        if not value:
            raise ValueError("Empty answer option.")
        options[m.group(1)] = value
    return options


def validate_bank(items, require_full=True):
    if require_full and len(items) != 35:
        raise ValueError("The final bank must contain all 35 original TSA items.")
    ids, texts = set(), set()
    for item in items:
        if item["id"] in ids:
            raise ValueError("Duplicate question ID.")
        ids.add(item["id"])
        if item["criterion"] not in CRITERIA:
            raise ValueError("Unrecognized TSA criterion.")
        if item["gold"] not in "ABCDE" or len(item["gold"]) != 1:
            raise ValueError("Invalid answer key.")
        fingerprint = digest(item["text"].encode("utf-8"))
        if fingerprint != item["question_sha256"]:
            raise ValueError(f"Question text changed: {item['id']}")
        if fingerprint in texts:
            raise ValueError("Duplicate question text.")
        texts.add(fingerprint)
        if list(option_block(item["text"])) != list("ABCDE"):
            raise ValueError("Each selected item must have five options.")
    if require_full:
        for criterion in CRITERIA:
            if sum(x["criterion"] == criterion for x in items) != 5:
                raise ValueError(f"Expected five items for {criterion}")


def parse_answer(text):
    """Never infer a choice from letters mentioned in a reason or a copied list."""
    clean = re.sub(r"<(think|thinking)\b[^>]*>.*?</\1\s*>", "",
                   text, flags=re.I | re.S).strip()
    if re.search(r"<(?:think|thinking)\b", clean, re.I):
        return {"choice": None, "status": "unclosed_thinking_tag"}
    clean = re.sub(r"(?m)^\s*#{1,6}\s*", "", clean)
    clean = clean.replace("**", "").replace(chr(96), "")
    pattern = (
        r"(?im)^[ \t]*(?:(?:answer|final(?:[ \t]+answer)?|correct[ \t]+answer)"
        r"[ \t]*:[ \t]*|(?:the[ \t]+)?correct[ \t]+answer[ \t]+is[ \t]+)"
        r"\(?([A-E])\)?(?:[.)])?(?=$|[ \t]+)([^\n]*)$"
    )
    candidates = re.findall(pattern, clean)
    if any(re.match(r"^\s*(?:or\s+[A-E]\b|[/,]\s*[A-E]\b)", tail, re.I)
           for _, tail in candidates):
        return {"choice": None, "status": "ambiguous_answer_marker"}
    matches = [letter for letter, _ in candidates]
    if len(set(m.upper() for m in matches)) > 1:
        return {"choice": None, "status": "conflicting_answer_markers"}
    if matches:
        return {"choice": matches[-1].upper(), "status": "explicit_answer_marker"}
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if lines and re.fullmatch(r"\(?[A-E]\)?[.)]?", lines[0], re.I):
        choice = re.search("[A-E]", lines[0], re.I).group().upper()
        return {"choice": choice, "status": "isolated_first_line"}
    return {"choice": None, "status": "no_unambiguous_answer"}


def wilson(correct, n):
    if not n:
        return None
    z = 1.959963984540054
    p, d = correct / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, center - half), min(1, center + half)]


def binomial_upper(correct, n, p=0.2):
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
               for k in range(correct, n + 1))


def validate_records(items, records):
    byid = {x["id"]: x for x in items}
    seen = set()
    for row in records:
        item_id = row.get("id")
        if item_id not in byid or item_id in seen:
            raise ValueError("Unexpected or duplicate response ID.")
        item = byid[item_id]
        if row["prompt"] != prompt_for(item):
            raise ValueError(f"Prompt mismatch: {item_id}")
        if not isinstance(row["response"], str):
            raise ValueError("Response must be raw text.")
        if row["parsed"] != parse_answer(row["response"]):
            raise ValueError(f"Stored extraction does not match raw response: {item_id}")
        seen.add(item_id)


def score(items, records):
    validate_records(items, records)
    byid = {x["id"]: x for x in records}
    results = []
    for criterion in CRITERIA:
        subset = [x for x in items if x["criterion"] == criterion]
        if not subset:
            continue
        rows = [byid[x["id"]] for x in subset if x["id"] in byid]
        correct = sum(byid[x["id"]]["parsed"]["choice"] == x["gold"]
                      for x in subset if x["id"] in byid)
        results.append({
            "criterion": criterion, "correct": correct, "maximum_points": len(subset),
            "completed": len(rows), "missing": len(subset) - len(rows),
            "invalid": sum(x["parsed"]["choice"] is None for x in rows),
            "accuracy": correct / len(subset) if len(rows) == len(subset) else None,
        })
    n = len(items)
    correct = sum(byid[x["id"]]["parsed"]["choice"] == x["gold"]
                  for x in items if x["id"] in byid)
    complete = len(records) == n
    predictions = {letter: sum(x["parsed"]["choice"] == letter for x in records)
                   for letter in ["A", "B", "C", "D", "E", None]}
    return {
        "completion": {"expected": n, "observed": len(records), "complete": complete,
                       "missing_ids": [x["id"] for x in items if x["id"] not in byid]},
        "answer_key_score": {
            "earned_points_so_far": correct, "maximum_points": n,
            "accuracy": correct / n if complete else None,
            "invalid": predictions[None],
            "prediction_counts": {str(k) if k is not None else "INVALID": v
                                  for k, v in predictions.items()},
            "wilson95_descriptive": wilson(correct, n) if complete else None,
            "uniform_guess_accuracy": 0.2,
            "best_constant_letter_accuracy": max(sum(x["gold"] == v for x in items)
                                                for v in "ABCDE") / n,
            "binomial_upper_tail_against_uniform_guessing":
                binomial_upper(correct, n) if complete else None,
            "note": "The binomial calculation assumes independent uniform guesses. "
                    "Public, selected items are not a random sample of all reasoning tasks.",
        },
        "criteria": results,
        "output_budget_hits": sum(x.get("finish_reason") == "length" for x in records),
        "seconds": sum(x.get("seconds", 0) for x in records),
        "explanation_review": {"status": "not_scored", "additional_TSA_points": 0},
        "limitations": [
            "35 original items from the study, in English; not the full study protocol.",
            "Seven measured argument-analysis skills; no global critical-thinking certification.",
            "One completion per item under one decoding configuration.",
            "Public question exposure during pretraining cannot be ruled out.",
            "Answer-key correctness and explanation quality are separate.",
            "Neither a correct answer nor a thinking tag proves internal process faithfulness.",
            "Five items per criterion give coarse, uncertain category estimates.",
            "Raw-score percentages are not Oxford admissions scaled scores or human percentiles.",
        ],
    }
