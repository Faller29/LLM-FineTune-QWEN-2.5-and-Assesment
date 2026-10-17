"""Shared primitives for a small CPU adaptation of NVIDIA's published SFT recipe."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
THINK = re.compile(r'<think>(.*?)</think>', re.I | re.S)
BOXED = re.compile(r'\\boxed\s*\{')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for raw in f:
            if raw.strip():
                yield json.loads(raw)


def save_jsonl(path, items):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def write_json(path, val):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(val, indent=2, ensure_ascii=False), encoding='utf-8')


def messages_of(record):
    value = record.get('input')
    if not isinstance(value, list):
        return []
    allowed = {'user', 'assistant', 'tool'}
    out = []
    for m in value:
        if isinstance(m, dict) and m.get('role') in allowed and isinstance(m.get('content'), str):
            out.append({'role': m['role'], 'content': m['content']})
    return out


def last_user(messages):
    return next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')


def nano_membership(record):
    value = record.get('used_in_training', [])
    if isinstance(value, (list, tuple, set)):
        return any(str(v).strip().lower() == 'nano' for v in value)
    return bool(re.search(r'\bNano\b', str(value), re.I))


def language_input(messages):
    return ' '.join(m['content'] for m in messages if m['role'] == 'user').replace('\n', ' ').strip()


def empty_thinking(output):
    # NVIDIA excludes records with an empty <think></think> section when mode is on.
    return any(not m.group(1).strip() for m in THINK.finditer(output))


def boxed_final(output, mode):
    # NVIDIA requires boxed final answers for math, not a box only in the trace.
    after = output
    if mode == 'on' and '</think>' in output.lower():
        after = re.split(r'</think>', output, flags=re.I)[-1]
    return bool(BOXED.search(after))


def boxed_value(text):
    matches = list(BOXED.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    balance, buf = 1, []
    for c in text[start:]:
        if c == '{': balance += 1
        elif c == '}':
            balance -= 1
            if balance == 0:
                return ''.join(buf).strip()
        buf.append(c)
    return None


def norm_question(value):
    return re.sub(r'\s+', ' ', value).strip().casefold()


def question_key(value):
    return hashlib.sha256(norm_question(value).encode('utf-8')).hexdigest()


def training_prompt(tokenizer, messages, mode):
    if mode not in ('on', 'off'):
        raise ValueError('reasoning must be on or off')
    # Same control phrases as NVIDIA's paper, but with the Qwen model's template.
    full = [{'role':'system', 'content':'detailed thinking '+mode}]
    full += [m for m in messages if m['role'] in ('user','assistant')]
    return tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=True)


def encoded_ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)['input_ids']


def training_tokens(tokenizer, row):
    prompt = encoded_ids(tokenizer, row['input'])
    completion = encoded_ids(tokenizer, row['output'] + tokenizer.eos_token)
    return prompt + completion, [-100] * len(prompt) + completion


def generate(model, tokenizer, prompt, max_new_tokens=128):
    import torch
    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated[0][encoded['input_ids'].size(1):], skip_special_tokens=True)


def test_overlap(splits):
    keys = [{r['question_key'] for r in group} for group in splits]
    return any(keys[i] & keys[j] for i in range(len(keys)) for j in range(i+1, len(keys)))


def count_groups(records):
    out = {}
    for r in records:
        k = r['category'] + '/' + r['reasoning']
        out[k] = out.get(k, 0) + 1
    return out
