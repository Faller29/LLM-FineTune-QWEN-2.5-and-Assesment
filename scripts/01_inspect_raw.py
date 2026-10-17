"""Stage 1: inspect the original NVIDIA JSONL files without filtering them."""
import argparse
import collections
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', nargs='+', default=['data/source/SFT/chat/chat.jsonl', 'data/source/SFT/math/math_v1.1.jsonl'])
p.add_argument('--limit-lines', type=int, default=2000, help='Sample inspection limit per input file, not curation')
p.add_argument('--show', type=int, default=2)
a = p.parse_args()
for fn in a.input:
    path = Path(fn)
    if not path.is_file():
        raise SystemExit(f'Missing source: {path}')
    count = collections.Counter()
    displayed = 0
    print('\nSOURCE:', path, 'size bytes:', path.stat().st_size)
    with path.open(encoding='utf-8') as f:
        for n, line in enumerate(f):
            if n >= a.limit_lines:
                break
            r = json.loads(line)
            count['read'] += 1
            count['reasoning/'+str(r.get('reasoning'))] += 1
            count['used_in_training/'+str(r.get('used_in_training'))] += 1
            if displayed < a.show:
                print('Example metadata:', {key:r.get(key) for key in ('reasoning','category','used_in_training','license')})
                print('Prompt preview:', str(r.get('input'))[:300])
                print('Output preview:', str(r.get('output'))[:300])
                displayed += 1
    print('Counts from inspected prefix:', dict(count))
