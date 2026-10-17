"""Ennis-Weir source import, integrity checks, and explicit reviewer scoring."""
from __future__ import annotations
import hashlib
import json
import math
import posixpath
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
SCENARIO_IDS = ["EW-ORIGINAL", "EW-NEW-1", "EW-NEW-2", "EW-NEW-3"]
FLAWS = {"EW-NEW-1": [2, 1, 2], "EW-NEW-2": [2, 1, 1], "EW-NEW-3": [1, 1, 1]}
CLOSING = {"overall_judgment": 1, "paragraph_summary": 1,
           "scope_overgeneralization": 2, "emotive_language": 1}
ERROR_KINDS = {"invented_flaw", "endorsed_flawed_argument", "faulty_reasoning",
               "unsupported_claim", "scope_omission", "other_manual_error"}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def signature(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line {line_number}; preserve this file before repair.") from exc
    return rows


def read_sheet(path, sheet_name):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = ["".join(t.text or "" for t in si.findall(".//m:t", ns))
                      for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        book = ET.fromstring(z.read("xl/workbook.xml"))
        sheet = next((x for x in book.findall("m:sheets/m:sheet", ns)
                      if x.get("name") == sheet_name), None)
        if sheet is None:
            raise ValueError(f"Missing worksheet: {sheet_name}")
        relations = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(x.get("Target") for x in relations
                      if x.get("Id") == sheet.get(f"{{{rel}}}id"))
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        rows = {}
        for row in ET.fromstring(z.read(member)).findall("m:sheetData/m:row", ns):
            cells = {}
            for cell in row.findall("m:c", ns):
                column = "".join(x for x in cell.get("r", "") if x.isalpha())
                value_node = cell.find("m:v", ns)
                value = value_node.text if value_node is not None else ""
                if cell.get("t") == "s":
                    value = shared[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(x.text or "" for x in cell.findall(".//m:t", ns))
                cells[column] = value
            rows[int(row.get("r"))] = cells
        return rows


def manual_paragraphs(path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    return ["".join(t.text or "" for t in p.findall(".//w:t", ns)).strip()
            for p in root.findall(".//w:p", ns)]


def split_manual(paragraphs):
    ordinal = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    original, new = {}, {}
    scenario, number = "EW-ORIGINAL", None
    for text in paragraphs:
        if text.startswith("Scoring Manual (newly"):
            scenario, number = None, None
        match = re.fullmatch(r"Scenario\s+(\d)\s*:?", text, re.I)
        if match:
            scenario, number = "EW-NEW-" + match[1], None
            continue
        if text.lower().startswith("paragraph ") and text.lower()[10:] in ordinal:
            number = ordinal.index(text.lower()[10:]) + 1
            (original if scenario == "EW-ORIGINAL" else new)[(scenario, number)] = []
            continue
        if number is not None and text:
            (original if scenario == "EW-ORIGINAL" else new)[(scenario, number)].append(text)
    if len(original) != 9 or len(new) != 9:
        raise ValueError("The scoring-manual paragraph layout differs from the verified release.")
    return original, new


def build_materials(workbook, manual):
    manifest = read_json(HERE / "ew_manifest.json")
    for key, path in [("workbook", workbook), ("manual", manual)]:
        if digest(Path(path).read_bytes()) != manifest["sources"][key]["sha256"]:
            raise ValueError(f"{key} SHA256 differs from the pinned published release.")
    old, new = split_manual(manual_paragraphs(manual))
    originals = read_sheet(workbook, "EW_original")
    additions = read_sheet(workbook, "EW_new")
    scenarios, units = [], []
    for sid in SCENARIO_IDS:
        original = sid == "EW-ORIGINAL"
        rows = originals if original else additions
        row_number = 2 if original else int(sid[-1]) + 1
        instruction, situation = rows[1]["B"], rows[row_number]["B"]
        prompt = instruction + "\n\n" + situation
        record = {"id": sid, "instruction": instruction, "situation": situation,
                  "prompt": prompt, "expected_response_paragraphs": 9 if original else 3,
                  "prompt_sha256": digest(prompt.encode("utf-8"))}
        reference_manifest = next(x for x in manifest["scenarios"] if x["id"] == sid)
        if record["prompt_sha256"] != reference_manifest["prompt_sha256"]:
            raise ValueError(f"Published prompt changed: {sid}")
        scenarios.append(record)
        for p in range(1, 10 if original else 4):
            contents = "\n".join((old if original else new)[(sid, p)])
            count = 1 if original else FLAWS[sid][p - 1]
            parts = re.split(r"\[Flaw\s*\d\]\s*", contents)
            for f in range(1, count + 1):
                reference = contents if count == 1 else parts[f]
                uid = f"{sid}-P{p}" + (f"-F{f}" if not original else "")
                units.append({"id": uid, "scenario_id": sid, "paragraph": p,
                              "flaw": None if original else f,
                              "maximum_points": 5 if original and p == 9 else 3,
                              "reference": reference})
    bank = {"scenarios": scenarios, "units": units, "provenance": manifest["sources"],
            "study_doi": manifest["study_doi"]}
    validate_bank(bank)
    return bank


def validate_bank(bank):
    if [x["id"] for x in bank["scenarios"]] != SCENARIO_IDS:
        raise ValueError("Expected the original and all three added scenarios.")
    if len(bank["units"]) != 21 or sum(x["maximum_points"] for x in bank["units"]) != 65:
        raise ValueError("Expected 21 scoring units and a nominal maximum of 65 points.")
    ids = [x["id"] for x in bank["units"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate scoring unit.")
    for scenario in bank["scenarios"]:
        if scenario["prompt"] != scenario["instruction"] + "\n\n" + scenario["situation"]:
            raise ValueError("Prompt contains content outside the released instruction and scenario.")
        if digest(scenario["prompt"].encode("utf-8")) != scenario["prompt_sha256"]:
            raise ValueError("Prompt fingerprint mismatch.")
        units = [x for x in bank["units"] if x["scenario_id"] == scenario["id"]]
        expected = 9 if scenario["id"] == "EW-ORIGINAL" else sum(FLAWS[scenario["id"]])
        if len(units) != expected or any(not x["reference"].strip() for x in units):
            raise ValueError("Incomplete reference/scoring map.")


def make_plan(bank, temperatures, rounds, seed, scope="full"):
    selected = bank["scenarios"] if scope == "full" else bank["scenarios"][:1]
    plan = []
    for temperature in temperatures:
        for round_number in range(1, rounds + 1):
            for scenario in selected:
                index = len(plan)
                plan.append({"id": f"{scenario['id']}-T{temperature:g}-R{round_number:03}",
                             "scenario_id": scenario["id"], "temperature": temperature,
                             "round": round_number, "generation_seed": seed + index})
    return plan


def validate_run(bank, config, records):
    validate_bank(bank)
    if signature(bank) != config["materials_sha256"]:
        raise ValueError("The saved scenarios/references changed after the run began.")
    plan, scenarios = config["plan"], {x["id"]: x for x in bank["scenarios"]}
    if len(records) > len(plan):
        raise ValueError("Too many saved responses.")
    for row, task in zip(records, plan):
        if any(row.get(k) != v for k, v in task.items()):
            raise ValueError("Response order or generation settings changed.")
        if row.get("prompt") != scenarios[task["scenario_id"]]["prompt"]:
            raise ValueError("Saved prompt differs from the complete published scenario.")
        if not isinstance(row.get("response"), str):
            raise ValueError("Raw response must be text.")
        if row.get("finish_reason") not in {"eos", "length", "other"}:
            raise ValueError("Missing generation completion status.")


def review_template(out, bank, records):
    cases = []
    for row in records:
        units = [x for x in bank["units"] if x["scenario_id"] == row["scenario_id"]]
        paragraphs = sorted(set(x["paragraph"] for x in units))
        ratings = [{"unit_id": x["id"], "status": "pending", "base_points": None,
                    "evidence": "", "reason": ""} for x in units]
        if row["scenario_id"] == "EW-ORIGINAL":
            ratings[-1]["closing_components"] = {
                k: {"credited": None, "evidence": "", "reason": ""} for k in CLOSING}
        cases.append({"response_id": row["id"], "ratings": ratings,
                      "paragraph_errors": [{"paragraph": p, "status": "pending", "errors": [],
                                            "scope_qualification_present": None,
                                            "comment": ""} for p in paragraphs]})
    return {"reviewer_name": "", "reviewer_type": "Manual",
            "run_config_sha256": digest((out / "run_config.json").read_bytes()),
            "responses_sha256": digest((out / "responses.jsonl").read_bytes()),
            "cases": cases}


def require_excerpt(text, excerpt, required):
    if not isinstance(excerpt, str) or (required and not excerpt.strip()):
        raise ValueError("Positive credit/explicit errors need evidence from the raw response.")
    if excerpt and excerpt not in text:
        raise ValueError("Review evidence is not an exact excerpt of the saved raw response.")


def score_review(out, bank, records, review):
    if review.get("run_config_sha256") != digest((out / "run_config.json").read_bytes()):
        raise ValueError("Review belongs to a different run configuration.")
    if review.get("responses_sha256") != digest((out / "responses.jsonl").read_bytes()):
        raise ValueError("Review belongs to different/older responses.")
    if not review.get("reviewer_name", "").strip() or review.get("reviewer_type") not in {"human", "Manual"}:
        raise ValueError("Record a reviewer name and type: human or Manual.")
    cases = review.get("cases", [])
    if len(cases) != len(records) or {c["response_id"] for c in cases} != {r["id"] for r in records}:
        raise ValueError("Review must cover each saved response exactly once.")
    if len({c["response_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate review response.")
    results = []
    byid = {r["id"]: r for r in records}
    for case in cases:
        row = byid[case["response_id"]]
        raw = row["response"]
        references = {x["id"]: x for x in bank["units"] if x["scenario_id"] == row["scenario_id"]}
        ratings = case["ratings"]
        if len(ratings) != len(references) or {x["unit_id"] for x in ratings} != set(references):
            raise ValueError("A review must retain every scoring unit, including omitted answers.")
        base, credited, complete = 0, 0, True
        for rating in ratings:
            if rating["status"] == "pending":
                if rating["base_points"] is not None:
                    raise ValueError("Pending ratings cannot contain points.")
                complete = False
                continue
            if rating["status"] != "reviewed":
                raise ValueError("Rating status must be pending or reviewed.")
            points = rating["base_points"]
            if type(points) is not int or not 0 <= points <= references[rating["unit_id"]]["maximum_points"]:
                raise ValueError("Base points must be an integer within the unit's published maximum.")
            require_excerpt(raw, rating.get("evidence", ""), points > 0)
            if not rating.get("reason", "").strip():
                raise ValueError("Every reviewed item requires a scoring explanation, including zeroes.")
            if references[rating["unit_id"]]["maximum_points"] == 5:
                parts = rating.get("closing_components", {})
                if set(parts) != set(CLOSING):
                    raise ValueError("The closing evaluation requires all four rubric components.")
                expected = 0
                for key, maximum in CLOSING.items():
                    part = parts[key]
                    if type(part.get("credited")) is not bool:
                        raise ValueError("Each closing component needs credited: true or false.")
                    require_excerpt(raw, part.get("evidence", ""), part["credited"])
                    if not part.get("reason", "").strip():
                        raise ValueError("Explain every closing component's judgment.")
                    expected += maximum if part["credited"] else 0
                if points != expected:
                    raise ValueError("Closing base score differs from its 1+1+2+1 components.")
            base += points
            credited += 1
        expected_paragraphs = {x["paragraph"] for x in references.values()}
        errors = case["paragraph_errors"]
        if len(errors) != len(expected_paragraphs) or {x["paragraph"] for x in errors} != expected_paragraphs:
            raise ValueError("Record errors once per response paragraph, not once per flaw.")
        manual_deduction, article_deduction = 0, 0
        for group in errors:
            if group["status"] == "pending":
                complete = False
                continue
            if group["status"] != "reviewed" or not group.get("comment", "").strip():
                raise ValueError("Each reviewed paragraph requires an error-review comment.")
            signatures = set()
            for error in group["errors"]:
                if error.get("kind") not in ERROR_KINDS or not error.get("explanation", "").strip():
                    raise ValueError("Each deduction requires an allowed kind and explanation.")
                omission = error.get("is_omission", False)
                if type(omission) is not bool:
                    raise ValueError("is_omission must be a boolean.")
                require_excerpt(raw, error.get("evidence", ""), not omission)
                key = (error["kind"], error.get("evidence", ""), error["explanation"])
                if key in signatures:
                    raise ValueError("Duplicate error instance within a paragraph.")
                signatures.add(key)
            count = len(group["errors"])
            if row["scenario_id"] == "EW-ORIGINAL" and group["paragraph"] in {3, 8}:
                qualification = group.get("scope_qualification_present")
                if type(qualification) is not bool:
                    raise ValueError("Original paragraphs 3 and 8 require an explicit scope-qualification judgment.")
                if not qualification and not any(e["kind"] == "scope_omission" for e in group["errors"]):
                    count += 1
            manual_deduction += min(1, count)
            article_deduction += count
        maximum = sum(x["maximum_points"] for x in references.values())
        results.append({"response_id": row["id"], "scenario_id": row["scenario_id"],
                        "temperature": row["temperature"], "round": row["round"],
                        "review_complete": complete, "rated_units": credited,
                        "expected_units": len(references), "maximum_points": maximum,
                        "base_alignment_points": base if complete else None,
                        "manual_capped_deductions": manual_deduction if complete else None,
                        "manual_score": base - manual_deduction if complete else None,
                        "article_per_error_deductions": article_deduction if complete else None,
                        "article_wording_sensitivity_score": base - article_deduction if complete else None})
    return {"reviewer_name": review["reviewer_name"], "reviewer_type": review["reviewer_type"],
            "cases": results}


def summarize_scores(config, reviewed):
    groups = {}
    expected = {}
    for task in config["plan"]:
        key = (task["temperature"], task["round"])
        expected.setdefault(key, set()).add(task["id"])
    for reviewer in reviewed:
        runs = []
        for (temperature, round_number), ids in expected.items():
            cases = [x for x in reviewer["cases"] if x["response_id"] in ids]
            complete = {x["response_id"] for x in cases} == ids and all(x["review_complete"] for x in cases)
            runs.append({"temperature": temperature, "round": round_number, "complete": complete,
                         "base_alignment_points": sum(x["base_alignment_points"] for x in cases) if complete else None,
                         "manual_score": sum(x["manual_score"] for x in cases) if complete else None,
                         "manual_capped_deductions": sum(x["manual_capped_deductions"] for x in cases) if complete else None,
                         "article_wording_sensitivity_score": sum(x["article_wording_sensitivity_score"] for x in cases) if complete else None})
        groups[reviewer["reviewer_name"]] = runs
    means = []
    for key in expected:
        matches = [next(x for x in runs if (x["temperature"], x["round"]) == key) for runs in groups.values()]
        complete = bool(matches) and all(x["complete"] for x in matches)
        means.append({"temperature": key[0], "round": key[1], "complete": complete,
                      "mean_manual_score": statistics.mean(x["manual_score"] for x in matches) if complete else None,
                      "mean_article_wording_sensitivity_score": statistics.mean(x["article_wording_sensitivity_score"] for x in matches) if complete else None})
    by_temperature = []
    for temperature in config["settings"]["temperatures"]:
        runs = [x for x in means if x["temperature"] == temperature]
        complete = all(x["complete"] for x in runs)
        scores = [x["mean_manual_score"] for x in runs if x["complete"]]
        by_temperature.append({"temperature": temperature, "complete": complete,
                               "rated_rounds": len(scores), "planned_rounds": len(runs),
                               "mean_score": statistics.mean(scores) if complete else None,
                               "sample_sd": statistics.stdev(scores) if complete and len(scores) > 1 else None})
    return {"individual_reviewer_rounds": groups, "mean_across_reviewers_by_round": means,
            "by_temperature": by_temperature,
            "all_ratings_complete": bool(reviewed) and all(x["complete"] for x in means),
            "inter_rater_reliability": "not_estimated",
            "note": "Reviewer averaging is descriptive. No human comparison, equating, or pass threshold is computed."}
