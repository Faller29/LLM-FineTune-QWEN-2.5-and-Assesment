"""Stage 4: collect ORIGINAL MODEL predictions on the same CPU-held-out test examples."""
import argparse
import json
import os
from pathlib import Path
from common import BASE_MODEL, boxed_value, generate, read_jsonl, write_json, save_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=BASE_MODEL)
    p.add_argument('--test',default='data/cpu/test.jsonl')
    p.add_argument('--limit',type=int,default=8)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',default='results/baseline.jsonl')
    a=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(max(1,a.threads))
    test=list(read_jsonl(a.test))
    if not test: raise SystemExit('No held-out examples. Run CPU selection first.')
    # Alternate math and chat if available; never score train data.
    groups={}
    for r in test: groups.setdefault((r['category'],r['reasoning']),[]).append(r)
    balanced=[]
    while len(balanced)<a.limit and any(groups.values()):
        for k in sorted(groups):
            if groups[k] and len(balanced)<a.limit:balanced.append(groups[k].pop(0))
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32).eval()
    preds=[]
    for ix,r in enumerate(balanced,1):
        text=generate(model,tokenizer,r['input'],a.max_new_tokens)
        preds.append({'question_key':r['question_key'], 'question':r['question'],
                      'reasoning':r['reasoning'], 'category':r['category'],
                      'reference':r['output'], 'base':text})
        print(f'{ix}/{len(balanced)} {r["category"]}/{r["reasoning"]}: {text[:180]!r}',flush=True)
    save_jsonl(a.output,preds)
    write_json(Path(a.output).with_suffix('.meta.json'),{'model':a.model,'test':a.test,
                'max_new_tokens':a.max_new_tokens,'rows':len(preds)})
    print('Saved',a.output)

if __name__=='__main__':main()
