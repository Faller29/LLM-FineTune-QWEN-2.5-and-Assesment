"""Evaluate ONE tuned model through the uploaded local web_app.py; stdlib only."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
import sys
import urllib.request
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parent
LABELS=['TRUE','FALSE','UNKNOWN']


def read_lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def extract(text,field,choices):
    # Ignore thought text. Require the requested marker on its own line.
    tail=re.split(r'</think>',text,flags=re.I)[-1]
    if '<think>' in tail.lower():
        return None
    found=re.findall(r'^\s*'+re.escape(field)+r'\s*:\s*('+'|'.join(choices)+r')\s*[.!]?\s*$',tail,re.I|re.M)
    return found[-1].upper() if found else None


def validate(cases):
    from build_suite import solve
    if len({x['id'] for x in cases})!=len(cases):
        raise ValueError('Duplicate case IDs')
    for x in cases:
        if x['task'] not in ('logic','critique','questions') or len(x['prompt'])>3000:
            raise ValueError('Unsupported task or oversized prompt')
        if x['task']!='questions':
            f=x['formal']
            gold=solve([tuple(v) for v in f['facts']],[(tuple(b),tuple(h)) for b,h in f['rules']],tuple(f['query']))
            if gold!=x['gold']:
                raise ValueError('Formal gold mismatch: '+x['id'])
            if x['task']=='critique' and (x['proposed']==gold)!=(x['gold_verdict']=='CORRECT'):
                raise ValueError('Critique gold mismatch: '+x['id'])


def request(base,route,body=None):
    p=urlsplit(base)
    if p.scheme!='http' or p.hostname not in ('localhost','127.0.0.1'):
        raise ValueError('Only the local HTTP model server is supported.')
    req=urllib.request.Request(base.rstrip('/')+route,
        data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=1800) as r:
        return json.load(r)


def select_cases(cases,smoke):
    # Reserve development families so final cases remain separate from debugging.
    if not smoke:
        return [x for x in cases if x['family'] not in ('L28','L29','L30','C19','C20','Q10')]
    ids=['L28_original','L29_original','L30_original','C19','C20','Q10']
    return [x for x in cases if x['id'] in ids]


def run(a):
    cases=read_lines(a.cases)
    validate(cases)
    selected=select_cases(cases,a.smoke)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    status=request(a.url,'/api/status')
    if not status.get('adapter_available'):
        raise ValueError('Start the server with your trained adapter; it is currently unavailable.')
    adapter=Path(status['adapter'])
    files={p.name:digest(p) for p in sorted(adapter.glob('*')) if p.is_file()}
    if not files:
        raise ValueError('Run this client on the model PC so the adapter can be fingerprinted.')
    config=dict(cases_sha256=digest(a.cases),client_sha256=digest(__file__),
        ids=[x['id'] for x in selected],mode=a.mode,max_new_tokens=a.max_new_tokens,
        model=status['model'],adapter_path=str(adapter),adapter_sha256=files,
        max_input=status['max_input'],protocol='single-model-custom-v1')
    cfg=out/'run_config.json'
    raw=out/'responses.jsonl'
    if cfg.exists() or raw.exists():
        if not a.resume:
            raise ValueError('Output exists. Choose a new --out or resume with identical settings.')
        if not cfg.exists() or json.loads(cfg.read_text(encoding='utf-8'))['config']!=config:
            raise ValueError('Cannot resume: model, adapter, suite, client, mode or budget changed.')
    else:
        write_json(cfg,dict(config=config,created_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
            note='Record the base-model/tokenizer revision, installed packages and hardware separately.'))
    old=read_lines(raw) if raw.exists() else []
    done={x['id'] for x in old}
    if len(done)!=len(old) or not done<=set(config['ids']):
        raise ValueError('Duplicate or unexpected saved response IDs.')
    random.Random(213).shuffle(selected)
    for case in selected:
        if case['id'] in done:
            continue
        answer=request(a.url,'/api/generate',dict(question=case['prompt'],task='chat',choice='tuned',
            mode=a.mode,history=[],max_new_tokens=a.max_new_tokens))
        if set(answer.get('outputs',{}))!={'tuned'}:
            raise ValueError('Unexpected server response; expected only tuned output.')
        output=answer['outputs']['tuned']
        record=dict(case,mode=a.mode,rendered_prompt=answer['prompt'],input_tokens=answer['input_tokens'],output=output)
        if case['task']=='logic':
            output['parsed']=extract(output['text'],'FINAL',LABELS)
        elif case['task']=='critique':
            output['parsed']=extract(output['text'],'CORRECTED',LABELS)
            output['verdict']=extract(output['text'],'VERDICT',['CORRECT','INCORRECT'])
        output['possible_budget_limit']=output['output_tokens']>=a.max_new_tokens
        with raw.open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
        done.add(case['id'])
        print(f"Saved {len(done)}/{len(config['ids'])}: {case['id']}",flush=True)
    print('Complete. Run summarize next.')


def wilson(k,n):
    if not n:
        return None
    z=1.95996398454
    p=k/n
    d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,center-half),min(1,center+half)]


def classification(rows):
    matrix={g:{p:0 for p in LABELS+['INVALID']} for g in LABELS}
    for x in rows:
        matrix[x['gold']][x['output']['parsed'] or 'INVALID']+=1
    correct=sum(matrix[g][g] for g in LABELS)
    per={}
    for label in LABELS:
        tp=matrix[label][label]
        fn=sum(matrix[label].values())-tp
        fp=sum(matrix[g][label] for g in LABELS if g!=label)
        per[label]=dict(support=tp+fn,precision=tp/(tp+fp) if tp+fp else 0,
            recall=tp/(tp+fn) if tp+fn else 0,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return dict(n=len(rows),correct=correct,accuracy=correct/len(rows) if rows else None,
        wilson95=wilson(correct,len(rows)),confusion_matrix=matrix,per_class=per,
        macro_f1=sum(x['f1'] for x in per.values())/3 if rows else None,
        invalid=sum(x['output']['parsed'] is None for x in rows),
        majority_baseline=max(Counter(x['gold'] for x in rows).values(),default=0)/len(rows) if rows else None)


def critique_scores(rows):
    tp=sum(x['gold_verdict']=='INCORRECT' and x['output']['verdict']=='INCORRECT' for x in rows)
    fp=sum(x['gold_verdict']=='CORRECT' and x['output']['verdict']=='INCORRECT' for x in rows)
    fn=sum(x['gold_verdict']=='INCORRECT' and x['output']['verdict']!='INCORRECT' for x in rows)
    tn=sum(x['gold_verdict']=='CORRECT' and x['output']['verdict']=='CORRECT' for x in rows)
    invalid=sum(x['output']['verdict'] is None for x in rows)
    bad=[x for x in rows if x['gold_verdict']=='INCORRECT']
    repaired=sum(x['output']['parsed']==x['gold'] for x in bad)
    return dict(n=len(rows),positive_class='INCORRECT',tp=tp,fp=fp,fn=fn,tn=tn,
        invalid_verdicts=invalid,verdict_accuracy=(tp+tn)/len(rows) if rows else None,
        error_precision=tp/(tp+fp) if tp+fp else 0,error_recall=tp/(tp+fn) if tp+fn else 0,
        error_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
        false_alarm_rate=fp/sum(x['gold_verdict']=='CORRECT' for x in rows) if any(x['gold_verdict']=='CORRECT' for x in rows) else None,
        incorrect_proposals=len(bad),corrected_classifications=repaired,
        correction_accuracy=repaired/len(bad) if bad else None,
        detection_and_correction_success=sum(x['output']['verdict']=='INCORRECT' and x['output']['parsed']==x['gold'] for x in bad),
        note='Correction scores judge the repaired truth label. Written explanation/repair validity is NOT automatically graded. Invalid judgments on correct cases are failures, not false alarms.')


def make_report(records,config,ratings=None):
    expected=set(config['ids'])
    observed={x['id'] for x in records}
    if len(observed)!=len(records) or not observed<=expected:
        raise ValueError('Duplicate/unexpected response IDs')
    logic=[x for x in records if x['task']=='logic']
    originals=[x for x in logic if x['variant']=='original']
    byvariant={v:classification([x for x in logic if x['variant']==v]) for v in ['original','paraphrase','changed']}
    families=defaultdict(dict)
    for x in logic:
        families[x['family']][x['variant']]=x
    complete=[f for f in families.values() if len(f)==3]
    allcorrect=sum(all(x['output']['parsed']==x['gold'] for x in f.values()) for f in complete)
    crit=critique_scores([x for x in records if x['task']=='critique'])
    questions=[x for x in records if x['task']=='questions']
    qscore=dict(responses=len(questions),human_review_required=True)
    if ratings is not None:
        allowed={'useful','unhelpful','invalid','missing'}
        qids={x['id'] for x in questions}
        if len({x['id'] for x in ratings})!=len(ratings) or {x['id'] for x in ratings}!=qids:
            raise ValueError('Ratings must contain each question-generation response exactly once.')
        marks=[]
        for x in ratings:
            if len(x.get('ratings',[]))!=3 or not set(x['ratings'])<=allowed:
                raise ValueError('Rate all three slots for each argument: useful/unhelpful/invalid/missing.')
            marks.extend(x['ratings'])
        counts=Counter(marks)
        qscore.update(dict(requested_slots=len(marks),counts=dict(counts),useful_fraction=counts['useful']/len(marks) if marks else None,
            note='Human ratings, not an automatic ground-truth score. Duplicate questions should receive no additional useful credit.'))
    return dict(model=config['model'],adapter=config['adapter_path'],mode=config['mode'],
        completion=dict(expected=len(expected),observed=len(observed),complete=observed==expected,missing=sorted(expected-observed)),
        logic=byvariant,logic_depth_originals={str(d):classification([x for x in originals if x['depth']==d])
            for d in [1,2,3,None]},
        robustness=dict(complete_families=len(complete),all_three_correct=allcorrect,
            all_three_correct_rate=allcorrect/len(complete) if complete else None),
        critique=crit,critical_questions=qscore,
        possible_budget_limit=sum(x['output']['possible_budget_limit'] for x in records),
        seconds=sum(x['output']['seconds'] for x in records),
        limitations=['Custom diagnostic, not official benchmark scores or a validated global critical-thinking scale.',
            'Default final set: 27 logic originals and 18 critique cases; limited exploratory evidence. Related variants are correlated.',
            'Wilson intervals describe uncertainty under independent-case sampling assumptions; this authored suite is not a random population sample.',
            'Format failures count as wrong; audit raw text for conclusions expressed outside the requested markers.',
            'F1 is classification F1, not token-overlap F1. Unsupported classes use zero_division=0.',
            'Written explanations are logged but not automatically judged. Correct labels do not prove internal reasoning faithfulness.',
            'Possible budget flag uses re-tokenized text, since the server supplies no definitive finish reason.'])


def review_html(records):
    cards=[]
    for x in records:
        if x['task']!='questions':
            continue
        opts='<option value="">Unrated</option>'+''.join(f'<option>{s}</option>' for s in ['useful','unhelpful','invalid','missing'])
        slots=''.join(f'<label>Question {i+1}: <select data-id="{x["id"]}" data-slot="{i}">{opts}</select></label> ' for i in range(3))
        cards.append('<article><h2>'+x['id']+'</h2><pre>'+html.escape(x['prompt'])+'</pre><h3>Model response</h3><pre>'+html.escape(x['output']['text'])+
            '</pre><details><summary>Reviewer guidance</summary><p>'+html.escape(x['reference'])+'</p><ul>'+''.join('<li>'+html.escape(q)+'</li>' for q in x['example_questions'])+
            '</ul><p>These are examples, not mandatory wording. Other valid, specific questions can be useful.</p></details><p>'+slots+'</p></article>')
    return '''<!doctype html><meta charset="utf-8"><title>Critical-question review</title>
<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-bottom:1px solid #aaa;padding:20px 0}select{padding:6px}button{padding:12px}</style>
<h1>Review the generated critical questions</h1><p>Useful: a valid, specific question whose answer could challenge this argument. Unhelpful: valid but does not meaningfully challenge it. Invalid: unrelated, faulty, overly generic or introduces unsupported content. Missing: no question supplied. Read only the first three questions; record fewer than three as missing. Give duplicate questions no additional useful credit (mark repeated valid questions unhelpful). The slot convention is a project adaptation.</p>
<p>Use reviewer guidance to judge meaning, not word overlap. Save each reviewer's ratings separately. The download contains no automatic truth judgments.</p>'''+''.join(cards)+'''<p><button onclick="downloadRatings()">Download ratings.json</button></p><script>
function downloadRatings(){const groups={};for(const s of document.querySelectorAll('select')){if(!s.value){alert('Rate every slot before downloading.');return;}if(!groups[s.dataset.id])groups[s.dataset.id]={id:s.dataset.id,ratings:[]};groups[s.dataset.id].ratings[Number(s.dataset.slot)]=s.value;}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(Object.values(groups),null,2)],{type:'application/json'}));a.download='ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
</script>'''


def summarize(a):
    out=Path(a.out)
    records=read_lines(out/'responses.jsonl')
    config=json.loads((out/'run_config.json').read_text(encoding='utf-8'))['config']
    ratings=json.loads(Path(a.ratings).read_text(encoding='utf-8')) if a.ratings else None
    report=make_report(records,config,ratings)
    filename='summary_with_human_ratings.json' if ratings is not None else 'summary.json'
    write_json(out/filename,report)
    (out/'critical_question_review.html').write_text(review_html(records),encoding='utf-8')
    audit=['# Written explanations: manual audit\n','These automatic scores do not grade explanation validity.\n']
    for x in records:
        if x['task']!='questions':
            audit.append('\n'+x['id']+'\nPROMPT:\n'+x['prompt']+'\nRESPONSE:\n'+x['output']['text']+'\nREFERENCE:\n'+x['reference']+'\n')
    (out/'explanation_audit.txt').write_text(''.join(audit),encoding='utf-8')
    print('Wrote '+filename+', critical_question_review.html and explanation_audit.txt')
    print('Logic originals:',report['logic']['original']['accuracy'],'; critique F1:',report['critique']['error_f1'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('validate')
    v.add_argument('--cases',type=Path,default=ROOT/'cases.jsonl')
    r=sub.add_parser('run')
    r.add_argument('--cases',type=Path,default=ROOT/'cases.jsonl')
    r.add_argument('--out',required=True)
    r.add_argument('--url',default='http://127.0.0.1:7860')
    r.add_argument('--mode',choices=['on','off'],default='on')
    r.add_argument('--max-new-tokens',type=int,choices=range(1,385),default=384,metavar='1..384')
    r.add_argument('--smoke',action='store_true')
    r.add_argument('--resume',action='store_true')
    s=sub.add_parser('summarize')
    s.add_argument('--out',required=True)
    s.add_argument('--ratings')
    a=p.parse_args()
    try:
        if a.command=='validate':
            cases=read_lines(a.cases)
            validate(cases)
            print('Validated',len(cases),'cases and all formal truth/critique labels. Human wording review is still required.')
        elif a.command=='run':
            run(a)
        else:
            summarize(a)
    except Exception as exc:
        raise SystemExit(f'{type(exc).__name__}: {exc}')


if __name__=='__main__':
    main()
