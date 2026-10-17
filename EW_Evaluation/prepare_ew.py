"""Import the authors' exact English EW scenarios and reference answers."""
from __future__ import annotations
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path
from ew_core import HERE, build_materials, digest, read_json, write_json


def acquire(name, source, override=None):
    destination = HERE / "sources" / source["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = [Path(override)] if override else [destination]
    if name == "workbook" and not override:
        candidates.append(HERE.parent / "TSA_Evaluation" / "sources" / "all_items.xlsx")
    for path in candidates:
        if path.is_file():
            blob = path.read_bytes()
            if digest(blob) != source["sha256"]:
                raise ValueError(f"Unexpected {name} edition at {path}. Use the pinned release.")
            destination.write_bytes(blob)
            return destination
    if override:
        raise ValueError(f"File not found: {override}")
    print(f"Downloading published {name}: {source['filename']}...", flush=True)
    request = urllib.request.Request(source["url"], headers={"User-Agent": "EW-academic-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        blob = response.read(5_000_001)
    if len(blob) > 5_000_000 or digest(blob) != source["sha256"]:
        raise ValueError(f"The downloaded {name} is not the verified published edition.")
    destination.write_bytes(blob)
    return destination


def prepare(workbook=None, manual=None):
    manifest = read_json(HERE / "ew_manifest.json")
    workbook_path = acquire("workbook", manifest["sources"]["workbook"], workbook)
    manual_path = acquire("manual", manifest["sources"]["manual"], manual)
    bank = build_materials(workbook_path, manual_path)
    write_json(HERE / "materials.json", bank)
    lines = ["PUBLISHED ENNIS-WEIR MODEL INPUTS", "No reference answers are included in these inputs.", ""]
    for scenario in bank["scenarios"]:
        lines += [scenario["id"], scenario["prompt"], ""]
    (HERE / "PROMPTS_FOR_INSPECTION.txt").write_text("\n".join(lines), encoding="utf-8")
    write_json(HERE / "REFERENCE_ANSWERS_DO_NOT_PROMPT.json", {"units": bank["units"]})
    print("Prepared 4 complete scenarios, 21 scoring units, nominal maximum 65 points.")
    print("Original letter: 9 units / 29 points. Added scenarios: 12 units / 36 points.")
    print("Reference answers remain outside all model prompts.")
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", help="Optional local copy of the pinned all_items.xlsx.")
    parser.add_argument("--manual", help="Optional local copy of the pinned Scoring Manual.docx.")
    args = parser.parse_args()
    try:
        prepare(args.workbook, args.manual)
        return 0
    except (OSError, ValueError, KeyError, IndexError, StopIteration) as exc:
        print(f"Preparation stopped: {exc}", file=sys.stderr)
        print("See README.txt for exact source URLs and offline-file fallback commands.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
