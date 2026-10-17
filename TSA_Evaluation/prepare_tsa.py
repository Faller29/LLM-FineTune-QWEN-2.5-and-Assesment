"""Download the published workbook and import the verified 35-item subset."""
from __future__ import annotations
import argparse
import json
import posixpath
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile
from tsa_core import digest, validate_bank, write_json

HERE = Path(__file__).resolve().parent
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def read_sheet(path, sheet_name):
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.findall(".//m:t", NS))
                      for si in root.findall("m:si", NS)]
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next((x for x in book.findall("m:sheets/m:sheet", NS)
                      if x.get("name") == sheet_name), None)
        if sheet is None:
            raise ValueError(f"Missing worksheet {sheet_name}")
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(x.get("Target") for x in relations
                      if x.get("Id") == sheet.get(f"{{{REL}}}id"))
        name = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        rows = {}
        for row in ET.fromstring(archive.read(name)).findall("m:sheetData/m:row", NS):
            cells = {}
            for cell in row.findall("m:c", NS):
                column = "".join(c for c in cell.get("r", "") if c.isalpha())
                value = cell.find("m:v", NS)
                value = value.text if value is not None else ""
                if cell.get("t") == "s":
                    value = shared[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(x.text or "" for x in cell.findall(".//m:t", NS))
                cells[column] = value
            rows[int(row.get("r"))] = cells
        return rows


def prepare(workbook=None):
    manifest = json.loads((HERE / "item_manifest.json").read_text(encoding="utf-8"))
    source = HERE / "sources" / "all_items.xlsx"
    source.parent.mkdir(parents=True, exist_ok=True)
    if workbook:
        data = Path(workbook).read_bytes()
    elif source.exists():
        data = source.read_bytes()
    else:
        print("Downloading the authors' workbook (about 46 KB)...", flush=True)
        request = urllib.request.Request(manifest["workbook_url"],
                                         headers={"User-Agent": "TSA-academic-evaluation/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(5_000_001)
        if len(data) > 5_000_000:
            raise ValueError("Unexpectedly large source file.")
    if digest(data) != manifest["workbook_sha256"]:
        raise ValueError("Workbook fingerprint differs from the verified edition. "
                         "Do not continue with silently changed questions.")
    source.write_bytes(data)
    cells = read_sheet(source, "TSA_original")
    items = []
    for record in manifest["items"]:
        row = cells[record["row"]]
        if row["A"] != record["criterion"]:
            raise ValueError(f"Criterion mismatch: {record['id']}")
        items.append({**record, "text": row["B"]})
    validate_bank(items)
    write_json(HERE / "questions.json", items)
    provenance = {k: v for k, v in manifest.items() if k != "items"}
    provenance["imported_ids"] = [x["id"] for x in items]
    write_json(HERE / "source_provenance.json", provenance)
    text = ["TSA ORIGINAL-ITEM SUBSET", "35 questions; five per criterion.",
            "This copy contains questions only. The runner sends no answer key to Qwen.", ""]
    for item in items:
        text += [f"{item['id']} | {item['criterion']} | "
                 f"TSA {item['year']} question {item['question_number']}", item["text"], ""]
    (HERE / "QUESTIONS_FOR_INSPECTION.txt").write_text("\n".join(text), encoding="utf-8")
    print("Prepared 35 verified questions, five per criterion.")
    print("Created questions.json and QUESTIONS_FOR_INSPECTION.txt.")
    print("The answer key is stored locally for scoring; it is never included in model prompts.")
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", help="Use your downloaded all_items.xlsx instead of downloading.")
    args = parser.parse_args()
    try:
        prepare(args.workbook)
    except (OSError, ValueError, KeyError, StopIteration, ET.ParseError,
            urllib.error.URLError) as exc:
        print(f"Preparation stopped: {exc}", file=sys.stderr)
        print("Manual fallback: download all_items.xlsx from the authors' repository, "
              "then run prepare_tsa.py --workbook PATH_TO_FILE.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
