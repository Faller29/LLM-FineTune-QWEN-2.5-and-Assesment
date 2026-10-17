"""Stage 2: NVIDIA article filtering; ONLY the tokenizer differs for Qwen2.5.

Reference: https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/
This is an independent CPU implementation of the article's described rules,
not NVIDIA NeMo Curator itself and not a byte-for-byte reproduction of its code.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

from common import (BASE_MODEL, boxed_final, count_groups, empty_thinking,
                    encoded_ids, language_input, last_user, messages_of,
                    nano_membership, question_key, save_jsonl, test_overlap,
                    training_prompt, write_json)


def curate(paths, tokenizer, lang_model, max_input=8192, max_completion=16384,
           max_per_source=1500, max_lines=0, seed=42, test_ratio=.10, val_ratio=.10):
    """Do not apply any 256/384-token curation filters or create synthetic examples."""
    stats = Counter()
    kept = []
    for file, category in paths:
        source = Path(file)
        if not source.is_file():
            raise FileNotFoundError(f'NVIDIA source not downloaded: {source}')
        print('CURATING', category, source, flush=True)
        kept_for_source = 0
        with source.open(encoding='utf-8') as stream:
            for rownum, line in enumerate(stream, 1):
                if max_lines > 0 and rownum > max_lines:
                    stats['stopped_scan_limit'] += 1
                    break
                if max_per_source > 0 and kept_for_source >= max_per_source:
                    stats['stopped_eligible_limit'] += 1
                    break
                stats['read'] += 1
                try:
                    r = json.loads(line)
                except ValueError:
                    stats['reject_json'] += 1
                    continue
                if not isinstance(r, dict):
                    stats['reject_json'] += 1
                    continue
                if not nano_membership(r):
                    stats['reject_non_nano'] += 1
                    continue
                messages = messages_of(r)
                question = last_user(messages)
                mode = r.get('reasoning')
                output = r.get('output')
                if not messages or not question or mode not in ('on','off') or not isinstance(output, str) or not output.strip():
                    stats['reject_invalid'] += 1
                    continue
                content = language_input(messages)
                try:
                    label, confidence = lang_model.predict(content.replace('\n',' ')[:20000], k=1)
                except Exception:
                    stats['reject_language_error'] += 1
                    continue
                if label[0] != '__label__en':
                    stats['reject_non_english_fasttext'] += 1
                    continue
                if category == 'math' and not boxed_final(output, mode):
                    stats['reject_unboxed_math_final'] += 1
                    continue
                if mode == 'on' and empty_thinking(output):
                    stats['reject_empty_think'] += 1
                    continue
                prompt = training_prompt(tokenizer, messages, mode)
                input_tokens = len(encoded_ids(tokenizer, prompt))
                output_tokens = len(encoded_ids(tokenizer, output + tokenizer.eos_token))
                if input_tokens > max_input:
                    stats['reject_input_tokens'] += 1
                    continue
                if output_tokens > max_completion:
                    stats['reject_completion_tokens'] += 1
                    continue
                kept.append({'input':prompt, 'output':output,
                             'question':question, 'question_key':question_key(question),
                             'category':category, 'reasoning':mode,
                             'input_tokens':input_tokens, 'output_tokens':output_tokens,
                             'source':str(source).replace('\\','/'),
                             'fasttext_language':'en',
                             'language_confidence':round(float(confidence[0]), 4)})
                stats['kept_'+mode] += 1
                kept_for_source += 1
                if stats['read'] % 10000 == 0:
                    print('Processed:', stats['read'], 'kept:',len(kept), flush=True)
    # Avoid leakage: assign ALL records with the same question to one split.
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in kept:
        groups[row['question_key']].append(row)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    partitions = {'train':[], 'val':[], 'test':[]}
    # Stratify by first group record's category/mode where possible.
    strata = defaultdict(list)
    for key in group_keys:
        first = groups[key][0]
        strata[(first['category'], first['reasoning'])].append(key)
    for keys in strata.values():
        n = len(keys)
        nt = round(n * test_ratio) if n >= 10 else 0
        nv = round(n * val_ratio) if n >= 10 else 0
        for idx, key in enumerate(keys):
            split = 'test' if idx < nt else 'val' if idx < nt + nv else 'train'
            partitions[split].extend(groups[key])
    def curriculum(records):
        # NVIDIA: split on/off, sort ascending completion length, interleave.
        buckets = {mode:sorted((r for r in records if r['reasoning'] == mode),
                               key=lambda r:(r['output_tokens'], r['input_tokens']))
                   for mode in ('on','off')}
        ordered=[]
        longer = max(len(buckets['on']),len(buckets['off']))
        for i in range(longer):
            for mode in ('on','off'):
                if i < len(buckets[mode]): ordered.append(buckets[mode][i])
        return ordered
    partitions = {k:curriculum(v) for k,v in partitions.items()}
    if test_overlap(list(partitions.values())):
        raise RuntimeError('Train/validation/test question leakage detected')
    report = {
        'source':'NVIDIA Llama-Nemotron-Post-Training-Dataset',
        'article_filters':{'nano_only':True, 'subsets':['math_v1.1','chat'],
                           'language':'FastText lid.176.ftz top predicted language English',
                           'math_boxed_final':True, 'exclude_empty_reasoning':True,
                           'max_input_tokens':max_input, 'max_completion_tokens':max_completion,
                           'apply_model_chat_template':True,
                           'mode_control':'detailed thinking on/off',
                           'curriculum':'on/off length sort then interleave'},
        'adaptations':{'tokenizer':getattr(tokenizer,'name_or_path','unknown'),
                       'model':getattr(tokenizer,'name_or_path',BASE_MODEL),
                       'implementation':'Python + FastText, not NeMo Curator',
                       'per_source_sampling_ceiling':max_per_source,
                       'scan_line_ceiling_per_source':max_lines,
                       'split_ratios':{'test':test_ratio,'validation':val_ratio},
                       'notes':'Sampling and splitting are separately reported; no extra token caps in curation.'},
        'filter_counts':dict(stats), 'curated_total':len(kept),
        'train':len(partitions['train']), 'val':len(partitions['val']), 'test':len(partitions['test']),
        'composition':{key:count_groups(val) for key,val in partitions.items()},
        'source_order_warning':'When a sample limit is used, this is a prefix of eligible samples, not a random sample.',
        'warning':None if len(kept)>=100 else 'Fewer than 100 curated records; training evidence will be weak.'
    }
    return partitions, report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--math',default='data/source/SFT/math/math_v1.1.jsonl')
    p.add_argument('--chat',default='data/source/SFT/chat/chat.jsonl')
    p.add_argument('--model',default=BASE_MODEL)
    p.add_argument('--lang-model',default='data/lid.176.ftz')
    p.add_argument('--max-token-count',type=int,default=8192)
    p.add_argument('--max-completion-token-count',type=int,default=16384)
    p.add_argument('--max-per-source',type=int,default=1500,
                   help='Sampling ceiling EACH for math and chat; 0 uses all eligible records')
    p.add_argument('--max-lines',type=int,default=0,
                   help='0 scans all source rows; nonzero is an EXPLICIT preview scan limit')
    p.add_argument('--test-ratio',type=float,default=.10)
    p.add_argument('--val-ratio',type=float,default=.10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-dir',default='data/curated')
    a=p.parse_args()
    if not Path(a.lang_model).is_file(): p.error('FastText model missing. See README Stage 1.')
    if not (0 <= a.test_ratio < 1 and 0 <= a.val_ratio < 1 and a.test_ratio+a.val_ratio < 1):
        p.error('Invalid test/validation ratios')
    from transformers import AutoTokenizer
    import fasttext
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    lang=fasttext.load_model(a.lang_model)
    splits, report=curate([(a.math,'math'),(a.chat,'chat')], tokenizer, lang,
                          a.max_token_count,a.max_completion_token_count,
                          a.max_per_source,a.max_lines,a.seed,a.test_ratio,a.val_ratio)
    root=Path(a.output_dir)
    for k,v in splits.items(): save_jsonl(root / (k+'.jsonl'), v)
    write_json(root/'curation_report.json',report)
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__ == '__main__': main()
