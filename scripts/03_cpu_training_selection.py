"""Stage 3: CPU-BUDGET SELECTION, AFTER full NVIDIA-style curation.

This is not an NVIDIA data-curation rule. Never silently truncate examples.
Keeps the NVIDIA-filtered corpus intact and writes explicit skip statistics.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from common import count_groups, read_jsonl, save_jsonl, write_json


def select(rows, budget, limit):
    counts=Counter()
    accepted=[]
    for r in rows:
        counts['inspected']+=1
        total=r['input_tokens']+r['output_tokens']
        if total > budget:
            counts['not_trainable_with_cpu_context']+=1
            continue
        if limit and len(accepted)>=limit:
            counts['eligible_not_selected_due_to_sample_limit']+=1
            continue
        accepted.append(r)
        counts['selected']+=1
    return accepted, counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--curated-dir',default='data/curated')
    p.add_argument('--output-dir',default='data/cpu')
    p.add_argument('--sequence-budget',type=int,default=512)
    p.add_argument('--train-limit',type=int,default=400)
    p.add_argument('--val-limit',type=int,default=32)
    p.add_argument('--test-limit',type=int,default=32)
    a=p.parse_args()
    if a.sequence_budget<64: p.error('sequence-budget must be >=64')
    report={'article_curation_unchanged':True,
            'cpu_sequence_budget':a.sequence_budget,
            'selection_is_hardware_adaptation_not_nvidia_filter':True,'splits':{}}
    for split,limit in [('train',a.train_limit),('val',a.val_limit),('test',a.test_limit)]:
        path=Path(a.curated_dir)/(split+'.jsonl')
        if not path.exists(): p.error('Missing curation file: '+str(path))
        selected,counts=select(read_jsonl(path),a.sequence_budget,limit)
        save_jsonl(Path(a.output_dir)/(split+'.jsonl'), selected)
        report['splits'][split]={'counts':dict(counts),'composition':count_groups(selected)}
        print(split,len(selected),'records |',dict(counts),'|',count_groups(selected))
    write_json(Path(a.output_dir)/'cpu_selection_report.json',report)
    if not report['splits']['train']['counts'].get('selected'):
        raise SystemExit('No examples fit this CPU context. Increase --sequence-budget carefully.')

if __name__ == '__main__':main()
