"""Local CPU model playground with no extra web dependencies.

Open http://127.0.0.1:7860 after running: python web_app.py
Only Python's standard-library HTTP server is used. The model still needs
PyTorch, Transformers, PEFT and the Hugging Face checkpoint.
"""
import argparse
import csv
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from common import BASE_MODEL, boxed_value, generate, training_prompt, encoded_ids

LOCK = threading.RLock()
MODEL = None
TOKENIZER = None
LOAD_ERROR = None
CONFIG = {}


def setup(adapter=None, model=BASE_MODEL, max_input=512,
          threads=4, output='results'):
    possible = [ROOT / 'results/qwen_lora/best', ROOT / 'results/qwen_lora/final']
    chosen = Path(adapter).expanduser() if adapter else next(
        (p for p in possible if (p / 'adapter_config.json').exists()), possible[0])
    if not chosen.is_absolute():
        chosen = ROOT / chosen
    CONFIG.clear()
    CONFIG.update(base_model=model, adapter=str(chosen), max_input=max_input,
                  threads=threads, output_dir=str(ROOT / output))


def adapter_exists():
    return (Path(CONFIG['adapter']) / 'adapter_config.json').exists()


def load_model():
    global MODEL, TOKENIZER, LOAD_ERROR
    if MODEL is not None:
        return MODEL, TOKENIZER
    with LOCK:
        if MODEL is not None:
            return MODEL, TOKENIZER
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_num_threads(min(CONFIG['threads'], max(1, os.cpu_count() or 1)))
            tok = AutoTokenizer.from_pretrained(CONFIG['base_model'])
            base = AutoModelForCausalLM.from_pretrained(
                CONFIG['base_model'], use_safetensors=True).eval()
            if adapter_exists():
                from peft import PeftModel
                loaded = PeftModel.from_pretrained(base, CONFIG['adapter']).eval()
            else:
                loaded = base
            TOKENIZER, MODEL = tok, loaded
            LOAD_ERROR = None
            return MODEL, TOKENIZER
        except Exception as exc:
            LOAD_ERROR = f'{type(exc).__name__}: {exc}'
            raise


def json_or_empty(path):
    try:
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    except (ValueError, OSError):
        return None


def lab_report():
    d = Path(CONFIG['output_dir'])
    c = json_or_empty(ROOT / 'data/curated/curation_report.json') or {}
    cpu = json_or_empty(ROOT / 'data/cpu/cpu_selection_report.json') or {}
    tr = json_or_empty(ROOT / 'results/qwen_lora/summary.json')
    ev = json_or_empty(ROOT / 'results/compare_metrics.json') or {}
    comp = ((cpu.get('splits') or {}).get('train') or {}).get('composition') or {}
    c_ui = dict(train_examples=((cpu.get('splits') or {}).get('train') or {}).get('counts',{}).get('selected',0),
                validation_examples=((cpu.get('splits') or {}).get('val') or {}).get('counts',{}).get('selected',0),
                test_examples=((cpu.get('splits') or {}).get('test') or {}).get('counts',{}).get('selected',0),
                train_category_counts={'math':sum(v for k,v in comp.items() if k.startswith('math/')),
                                       'chat':sum(v for k,v in comp.items() if k.startswith('chat/'))},
                train_mode_counts={'on':sum(v for k,v in comp.items() if k.endswith('/on')),
                                   'off':sum(v for k,v in comp.items() if k.endswith('/off'))})
    box = ev.get('math_boxed') or {}
    denominator = box.get('scorable',0)
    ev_ui = dict(evaluated=ev.get('records',0),
                 boxed_math_accuracy={'base':box.get('base',0)/denominator if denominator else None,
                                      'fine_tuned':box.get('tuned',0)/denominator if denominator else None})
    return {'curation':c_ui, 'original_curation':c, 'cpu_selection':cpu, 'training':tr,
            'training_config':json_or_empty(ROOT / 'results/qwen_lora/training_config.json'),
            'evaluation':ev_ui, 'evaluation_details':ev}



def format_input(tokenizer, question, mode, task, history):
    content = question.strip()
    if task == 'math' and '\\boxed' not in content and 'put the answer' not in content.lower():
        content = ('Solve the following math problem. Make sure to put the '
                   'answer (and only answer) inside \\boxed{}.\n\n' + content)
    messages=[]
    for turn in history[-2:]:
        messages.append({'role':'user','content':turn['user'].strip()})
        messages.append({'role':'assistant','content':turn['assistant'].strip()})
    messages.append({'role':'user','content':content})
    return training_prompt(tokenizer, messages, mode)


def status_data():
    return dict(model=CONFIG['base_model'], adapter=CONFIG['adapter'],
                adapter_available=adapter_exists(), model_loaded=MODEL is not None,
                model_error=LOAD_ERROR, max_input=CONFIG['max_input'], cpu_only=True)


def loss_data():
    root = Path(CONFIG['output_dir'])
    candidate = root / 'qwen_lora/loss.csv'
    if not candidate.exists():
        return dict(rows=[], message='No training measurements yet')
    with candidate.open(newline='', encoding='utf-8') as f:
        measurements = list(csv.DictReader(f))
    return dict(rows=measurements, source=str(candidate))


def examples_data():
    file = ROOT / 'data/cpu/test.jsonl'
    groups = {}
    if file.is_file():
        try:
            with file.open(encoding='utf-8') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    example = json.loads(line)
                    label = (str(example.get('category')), str(example.get('reasoning')))
                    question = str(example.get('question') or '')
                    if question and len(question) <= 900 and label not in groups:
                        groups[label] = dict(question=question, category=label[0], reasoning=label[1])
        except (OSError, ValueError):
            return dict(examples=[])
    return dict(examples=list(groups.values()))


def run_inference(body):
    """Return a (JSON response, HTTP status) pair for offline tests and HTTP."""
    if not isinstance(body, dict):
        return dict(error='Expected a JSON object.'), 400
    question = str(body.get('question') or '').strip()
    mode, task, choice = body.get('mode', 'on'), body.get('task', 'math'), body.get('choice', 'compare')
    if not question or len(question) > 3000:
        return dict(error='Enter a question with at most 3,000 characters.'), 400
    if mode not in ('on', 'off') or task not in ('math', 'chat') or choice not in ('base', 'tuned', 'compare'):
        return dict(error='Invalid reasoning mode, task, or model choice.'), 400
    try:
        max_new = int(body.get('max_new_tokens', 128))
    except (TypeError, ValueError):
        return dict(error='max_new_tokens must be a number.'), 400
    if not 1 <= max_new <= 384:
        return dict(error='Select between 1 and 384 output tokens.'), 400
    history = body.get('history') or []
    if not isinstance(history, list) or len(history) > 3 or any(
            not isinstance(t, dict) or
            not isinstance(t.get('user'), str) or not isinstance(t.get('assistant'), str) or
            len(t['user']) > 2000 or len(t['assistant']) > 2500 for t in history):
        return dict(error='Invalid conversation history; clear the chat and retry.'), 400
    with LOCK:
        try:
            if choice in ('tuned', 'compare') and not adapter_exists():
                return dict(error='LoRA adapter not found. Train first or choose Base model.'), 400
            model, tok = load_model()
            prompt = format_input(tok, question, mode, task, history)
            in_tokens = len(encoded_ids(tok, prompt))
            if in_tokens > CONFIG['max_input']:
                return dict(error=f'Input has {in_tokens} tokens but the model limit is '
                                  f'{CONFIG["max_input"]}. Shorten it or clear chat history.'), 400
            answers = {}
            for name in (['base', 'tuned'] if choice == 'compare' else [choice]):
                started = time.perf_counter()
                if name == 'base' and adapter_exists():
                    with model.disable_adapter():
                        response = generate(model, tok, prompt, max_new_tokens=max_new)
                else:
                    response = generate(model, tok, prompt, max_new_tokens=max_new)
                answers[name] = dict(text=response, boxed_answer=boxed_value(response),
                                     seconds=round(time.perf_counter() - started, 2),
                                     output_tokens=len(tok(response, add_special_tokens=False)['input_ids']))
        except (RuntimeError, OSError, ImportError, ValueError) as exc:
            return dict(error=f'{type(exc).__name__}: {exc}'), 500
    return dict(prompt=prompt, input_tokens=in_tokens, mode=mode,
                task=task, outputs=answers, adapter_available=adapter_exists()), 200


class LabHandler(BaseHTTPRequestHandler):
    def send_bytes(self, data, content_type='application/octet-stream', status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data, status=200):
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode('utf-8'),
                        'application/json; charset=utf-8', status)

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == '/api/status':
            return self.send_json(status_data())
        if route == '/api/experiment':
            return self.send_json(lab_report())
        if route == '/api/loss':
            return self.send_json(loss_data())
        if route == '/api/examples':
            return self.send_json(examples_data())
        if route == '/':
            path = ROOT / 'web/templates/index.html'
        elif route in ('/static/style.css', '/static/app.js',
                       '/static/format.js', '/static/vendor/tex-svg.js'):
            path = ROOT / 'web' / route.lstrip('/')
        else:
            return self.send_json(dict(error='Not found'), 404)
        try:
            kind = mimetypes.guess_type(str(path))[0] or 'text/plain'
            return self.send_bytes(path.read_bytes(), kind + ('; charset=utf-8' if kind.startswith('text/') or 'javascript' in kind else ''))
        except OSError as exc:
            return self.send_json(dict(error=str(exc)), 500)

    def do_POST(self):
        if urlsplit(self.path).path != '/api/generate':
            return self.send_json(dict(error='Not found'), 404)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > 65536:
                return self.send_json(dict(error='Invalid request size'), 413)
            if 'application/json' not in self.headers.get('Content-Type', ''):
                return self.send_json(dict(error='JSON content required'), 400)
            body = json.loads(self.rfile.read(length))
        except (ValueError, TypeError, UnicodeDecodeError):
            return self.send_json(dict(error='Invalid JSON body'), 400)
        response, status = run_inference(body)
        return self.send_json(response, status)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', help='Path to LoRA adapter (default: results/qwen_lora/best)')
    p.add_argument('--model', default=BASE_MODEL)
    p.add_argument('--max-input', type=int, default=512)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--port', type=int, default=7860)
    p.add_argument('--host', default='127.0.0.1')
    a = p.parse_args()
    setup(a.adapter, a.model, a.max_input, a.threads)
    print(f'Open http://{a.host}:{a.port}/ in your browser.')
    print('No web-service dependencies. Model weights are loaded at the first request.', flush=True)
    with ThreadingHTTPServer((a.host, a.port), LabHandler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('Stopping the lab server.')
