"""Stage 5: CPU PyTorch LoRA SFT on NVIDIA-filtered CPU-feasible records.

Decoder-only labels mask the prompt; only the target continuation contributes loss.
Uses native Qwen chat prompts saved by Stage 2 and NO silent token truncation.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import time
from common import BASE_MODEL, encoded_ids, read_jsonl, training_tokens, write_json


def sample_validation_loss(model, examples, torch):
    if not examples:return None
    model.eval()
    sum_loss, sum_tokens=0.,0
    with torch.no_grad():
        for ids,labels in examples:
            x=torch.tensor([ids],dtype=torch.long)
            y=torch.tensor([labels],dtype=torch.long)
            loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y).loss
            n=sum(v!=-100 for v in labels)
            sum_loss+=float(loss)*n
            sum_tokens+=n
    model.train()
    return sum_loss/max(sum_tokens,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=BASE_MODEL)
    p.add_argument('--train',default='data/cpu/train.jsonl')
    p.add_argument('--validation',default='data/cpu/val.jsonl')
    p.add_argument('--output',default='results/qwen_lora')
    p.add_argument('--max-steps',type=int,default=60)
    p.add_argument('--grad-accumulation',type=int,default=4)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--alpha',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--max-val',type=int,default=8)
    p.add_argument('--max-sequence',type=int,default=512)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if min(a.max_steps,a.grad_accumulation,a.rank,a.alpha,a.max_sequence,a.eval_every,a.threads) <= 0:
        p.error('Positive counts are required')
    import torch
    from transformers import (AutoTokenizer,AutoModelForCausalLM,
                              get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, TaskType
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(a.threads)
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    training=list(read_jsonl(a.train))
    validation=list(read_jsonl(a.validation)) if Path(a.validation).exists() else []
    if not training: raise SystemExit('Empty train set. Run scripts/03_cpu_training_selection.py')
    train_keys={r['question_key'] for r in training}
    if train_keys.intersection(r['question_key'] for r in validation):
        raise SystemExit('Train/validation question leakage.')
    def encode_all(items):
        output=[]
        for r in items:
            ids,labels=training_tokens(tokenizer,r)
            if len(ids)>a.max_sequence:
                raise SystemExit(f'Example too long ({len(ids)} > {a.max_sequence}); re-run stage 3 with same budget')
            output.append((ids,labels))
        return output
    encoded=encode_all(training)
    encoded_val=encode_all(validation[:a.max_val])
    print('Train:',len(encoded),'Validation:',len(encoded_val),'CPU max sequence:',a.max_sequence)
    base=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32)
    base.config.use_cache=False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
                      r=a.rank, lora_alpha=a.alpha, lora_dropout=.05,
                      target_modules=['q_proj','v_proj']))
    model.print_trainable_parameters()
    model.train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=a.lr,weight_decay=.001)
    scheduler=get_cosine_schedule_with_warmup(optimizer,
              num_warmup_steps=max(1,math.ceil(.05*a.max_steps)),
              num_training_steps=a.max_steps)
    result=Path(a.output); result.mkdir(parents=True,exist_ok=True)
    config=vars(a).copy()
    config.update(training_rows=len(encoded),validation_rows=len(encoded_val),
                  note='Not NVIDIA NeMo training: CPU LoRA on Qwen2.5 with explicit train-time context budget')
    write_json(result/'training_config.json', config)
    validation_best=float('inf');best_step=None
    ema=None
    seen=0
    started=time.perf_counter()
    # Loop through curriculum once then repeat; a new pass starts from easiest.
    cursor=0
    with (result/'loss.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(['step','loss','ema_loss','val_loss','learning_rate','examples_seen','seconds'])
        for step in range(1,a.max_steps+1):
            optimizer.zero_grad(set_to_none=True)
            losses=[]
            for micro in range(a.grad_accumulation):
                ids,labels=encoded[cursor%len(encoded)]
                cursor+=1
                x=torch.tensor([ids],dtype=torch.long)
                y=torch.tensor([labels],dtype=torch.long)
                loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y).loss
                (loss/a.grad_accumulation).backward()
                losses.append(float(loss.detach()))
                seen+=1
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
            optimizer.step(); scheduler.step()
            current=sum(losses)/len(losses)
            ema=current if ema is None else .9*ema+.1*current
            val=None
            if encoded_val and (step % a.eval_every==0 or step==a.max_steps):
                val=sample_validation_loss(model,encoded_val,torch)
                if val<validation_best:
                    validation_best=val;best_step=step
                    model.save_pretrained(result/'best')
            writer.writerow([step,round(current,6),round(ema,6),
                             round(val,6) if val is not None else '',
                             optimizer.param_groups[0]['lr'],seen,
                             round(time.perf_counter()-started,2)])
            file.flush()
            print(f'Step {step}/{a.max_steps}: loss={current:.3f} ema={ema:.3f}'+
                  (f' val={val:.3f}' if val is not None else ''),flush=True)
    model.save_pretrained(result/'final')
    write_json(result/'summary.json',{'steps':a.max_steps,'examples_seen':seen,
                 'best_validation_loss':None if best_step is None else validation_best,
                 'best_step':best_step,'elapsed_seconds':round(time.perf_counter()-started,2),
                 'model':a.model})
    print('Saved:',result/'best','and',result/'final')

if __name__=='__main__':main()
