"""Stage 6: compare original model with LoRA on identical held-out test examples."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
from common import BASE_MODEL, boxed_value, generate, read_jsonl, save_jsonl, write_json


def score(records):
    groups=Counter()
    box=Counter()
    fmt=Counter()
    for r in records:
        groups[r['category']+'/'+r['reasoning']]+=1
        if r['category']=='math':
            correct=boxed_value(r['reference'])
            if correct is not None:
                box['scorable']+=1
                for key in ['base','tuned']:
                    if boxed_value(r[key])==correct: box[key]+=1
        for key in ['base','tuned']:
            output=r[key]
            good=('<think>' in output and '</think>' in output) if r['reasoning']=='on' else '<think>' not in output
            if good:fmt[key]+=1
    return {'records':len(records),'categories':dict(groups),
            'math_boxed':{'scorable':box['scorable'],
                          'base':box['base'],'tuned':box['tuned'],
                          'note':'Exact boxed-string match, not general mathematical correctness'},
            'reasoning_format':{'base':fmt['base'],'tuned':fmt['tuned'],
                                'denominator':len(records)},
            'limitations':['Model is 0.49B parameters, not NVIDIA 8B.',
                           'NVIDIA GPQA/MMLU benchmark values are not replicated.',
                           'Thought tags show output format, not correctness of thought.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=BASE_MODEL)
    p.add_argument('--adapter',default='results/qwen_lora/best')
    p.add_argument('--test',default='data/cpu/test.jsonl')
    p.add_argument('--limit',type=int,default=8)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--output',default='results/compare')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    if not (Path(a.adapter)/'adapter_config.json').exists():
        raise SystemExit('Adapter not found. Train first: '+a.adapter)
    torch.set_num_threads(a.threads)
    base=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32)
    tuned=PeftModel.from_pretrained(base,a.adapter).eval()
    tok=AutoTokenizer.from_pretrained(a.model)
    test=list(read_jsonl(a.test))
    groups={}
    for r in test:groups.setdefault((r['category'],r['reasoning']),[]).append(r)
    balanced=[]
    while len(balanced)<a.limit and any(groups.values()):
        for k in sorted(groups):
            if len(balanced)>=a.limit:break
            if groups[k]: balanced.append(groups[k].pop(0))
    result=[]
    for n,row in enumerate(balanced,1):
        with tuned.disable_adapter():
            original=generate(tuned,tok,row['input'],a.max_new_tokens)
        adapted=generate(tuned,tok,row['input'],a.max_new_tokens)
        result.append({'question_key':row['question_key'],'question':row['question'],
                       'category':row['category'],'reasoning':row['reasoning'],
                       'reference':row['output'],'base':original,'tuned':adapted})
        print(f'[{n}/{len(balanced)}] base={original[:100]!r} tuned={adapted[:100]!r}',flush=True)
    root=Path(a.output)
    save_jsonl(str(root)+'_predictions.jsonl',result)
    metrics=score(result)
    write_json(str(root)+'_metrics.json',metrics)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
