"""Create ORIGINAL diagnostic cases; not copies of published benchmarks.
Logic keys are checked by a forward-chaining engine with explicit negation.
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
LABELS = ('TRUE', 'FALSE', 'UNKNOWN')


def solve(facts, rules, query):
    known = set(facts)
    while True:
        extra = {head for body, head in rules if body in known} - known
        if not extra:
            break
        known.update(extra)
    opposite = (query[0], query[1], not query[2])
    if query in known and opposite in known:
        raise ValueError('Contradictory generated problem')
    return 'TRUE' if query in known else 'FALSE' if opposite in known else 'UNKNOWN'


def phrase(literal, paraphrase=False):
    entity, adjective, positive = literal
    if paraphrase:
        return f'{entity} has the property {adjective}' if positive else f'{entity} does not have the property {adjective}'
    return f'{entity} is {adjective}' if positive else f'{entity} is not {adjective}'


def problem(facts, rules, query, paraphrase=False):
    f = '\n'.join(f'F{i}: {phrase(x, paraphrase)}.' for i,x in enumerate(facts,1))
    r = '\n'.join(f'R{i}: If {phrase(body, paraphrase)}, then {phrase(head, paraphrase)}.'
        for i,(body,head) in enumerate(rules,1))
    return ('Use only the stated facts and forward, one-way rules. Treat absence of evidence as UNKNOWN, not FALSE. '
        'TRUE means the query is supported; FALSE means its explicit negation is supported.\n'
        f'Facts:\n{f}\nRules:\n{r}\nQuery: {phrase(query, paraphrase)}.')


def make_suite():
    rng = random.Random(2132026)
    adjectives = ['calm','bright','kind','quiet','round','blue','rough','young','soft','tall','cold','green']
    names = ['Neri','Mora','Tavi','Luma','Rivo','Sena','Daro','Vela','Kori','Panu']
    cases=[]
    originals=[]
    for i in range(30):
        label=LABELS[i%3]
        depth=1+(i//3)%3
        name=names[i%10]+str(i+1)
        attrs=rng.sample(adjectives,depth+3)
        literals=[(name,a,True) for a in attrs[:depth+1]]
        if label=='FALSE':
            literals[-1]=(name,literals[-1][1],False)
        rules=list(zip(literals,literals[1:]))
        query=(name,attrs[depth],True)
        if label=='UNKNOWN':
            # Knowing the end of a one-way chain does not establish its start.
            facts=[literals[-1]]
            query=literals[0]
        else:
            facts=[literals[0]]
        facts.append(('Other'+str(i),attrs[-1],True))
        rng.shuffle(rules)
        rng.shuffle(facts)
        assert solve(facts,rules,query)==label
        reason=('The forward chain supports the query.' if label=='TRUE' else
            'The forward chain supports the explicit negation of the query.' if label=='FALSE' else
            'Only the end of the one-way chain is known. Its reverse cannot establish the query or its negation.')
        fid=f'L{i+1:02d}'
        original=dict(id=fid+'_original',family=fid,task='logic',variant='original',
            prompt=problem(facts,rules,query)+'\nExplain briefly. End with FINAL: TRUE, FINAL: FALSE, or FINAL: UNKNOWN.',
            gold=label,reference=reason,depth=depth if label!='UNKNOWN' else None,
            formal=dict(facts=facts,rules=rules,query=query))
        originals.append(original)
        cases.append(original)
        cases.append(dict(original,id=fid+'_paraphrase',variant='paraphrase',
            prompt=problem(facts,rules,query,True)+'\nExplain briefly. End with FINAL: TRUE, FINAL: FALSE, or FINAL: UNKNOWN.'))
        if label=='UNKNOWN':
            changed_facts=facts+[query]
            changed_query=query
        else:
            changed_facts=facts
            changed_query=(query[0],query[1],not query[2])
        changed_gold=solve(changed_facts,rules,changed_query)
        assert changed_gold!=label
        cases.append(dict(original,id=fid+'_changed',variant='changed',gold=changed_gold,
            reference='The query polarity changed.' if label!='UNKNOWN' else 'The query was added as an explicit fact.',
            formal=dict(facts=changed_facts,rules=rules,query=changed_query),
            prompt=problem(changed_facts,rules,changed_query)+'\nExplain briefly. End with FINAL: TRUE, FINAL: FALSE, or FINAL: UNKNOWN.'))
    # Different entity names from logic families; independent critique records.
    for i in range(20):
        original=originals[(i*7)%30]
        formal=json.loads(json.dumps(original['formal']))
        old=formal['query'][0]
        new='Critique'+str(i+1)
        def rename(lit):
            return (new if lit[0]==old else lit[0],lit[1],lit[2])
        facts=[rename(x) for x in formal['facts']]
        rules=[(rename(b),rename(h)) for b,h in formal['rules']]
        query=rename(formal['query'])
        gold=solve(facts,rules,query)
        correct=i%2==0
        proposed=gold if correct else LABELS[(LABELS.index(gold)+1)%3]
        cases.append(dict(id=f'C{i+1:02d}',family=f'C{i+1:02d}',task='critique',variant='original',
            gold=gold,gold_verdict='CORRECT' if correct else 'INCORRECT',proposed=proposed,
            reference=original['reference'],formal=dict(facts=facts,rules=rules,query=query),
            prompt=problem(facts,rules,query)+f'\nA proposed answer classifies the query as {proposed}. '
                'Judge this proposed conclusion. Explain any error briefly, then give the justified classification. '
                'End with these two lines:\nVERDICT: CORRECT or INCORRECT\nCORRECTED: TRUE, FALSE, or UNKNOWN'))
    arguments=[
        ('Five volunteers liked a training course, so it will improve every employee\'s job performance.',
         'Enjoyment is not measured job performance; volunteers may not represent all employees.',
         ['Was job performance measured against a suitable comparison group?','Are the volunteers representative of the employees?','Could other factors account for any performance changes?']),
        ('Our sales rose after the website redesign, so the redesign alone caused the increase.',
         'Temporal association alone does not isolate causation.',
         ['Were advertising or pricing changes controlled?','Was there a comparable group without the redesign?','Was the rise already part of a prior trend?']),
        ('A famous actor says software S is secure, so its security has been established.',
         'Popularity is not technical security evidence.',
         ['What relevant security expertise does the actor have?','Have independent security tests been performed?','What evidence supports the specific security claim?']),
        ('One customer had no problems with product P, so nobody will ever have a problem with it.',
         'One case does not justify a universal future guarantee.',
         ['How many customers were tested?','Were different usage conditions included?','Are there documented failures outside this one case?']),
        ('Either we buy a new system today or our work will stop forever; no other option is worth considering.',
         'The claim assumes an exhaustive two-option choice without evidence.',
         ['Could the current system be repaired?','Is a temporary alternative feasible?','What evidence establishes permanent stoppage without immediate purchase?']),
        ('An undated screenshot says a company policy is active, so it must be active today.',
         'Current validity and provenance are unverified.',
         ['When was the screenshot created?','Is there a current authenticated policy record?','Could the policy have been revised since the screenshot?']),
        ('A study of one branch found method M useful, so it must work equally well in every branch.',
         'Generalization requires evidence about different branch conditions.',
         ['Were other branches included?','Do branches differ in conditions relevant to M?','Is there replicated evidence outside the original branch?']),
        ('A forecast predicts Tuesday delivery, therefore Tuesday delivery is already a confirmed fact.',
         'A prediction is being treated as an observation.',
         ['Is there a delivery confirmation?','What uncertainty does the forecast report?','Has the predicted event actually occurred?']),
        ('We surveyed only the customers who renewed their subscriptions, and all were happy; therefore all customers are happy.',
         'The sample excludes non-renewing customers and may be systematically biased.',
         ['Were customers who did not renew surveyed?','Does the sample represent all customers?','Could dissatisfaction influence the decision not to renew?']),
        ('No one in our small team has disproved claim K, so claim K has been proved true.',
         'Absence of disproof is not sufficient positive proof.',
         ['What positive evidence supports K?','Was K tested adequately?','Could the team have lacked the evidence or resources needed to test K?'])]
    for i,(argument,weakness,examples) in enumerate(arguments,1):
        cases.append(dict(id=f'Q{i:02d}',family=f'Q{i:02d}',task='questions',variant='original',
            prompt='Read the argument below. Generate exactly THREE specific questions that could expose unsupported assumptions, '
                'weak evidence, or relevant alternative explanations. Do not answer the questions.\nArgument: '+argument,
            reference=weakness,example_questions=examples))
    for c in cases:
        c['provenance']='Original authored diagnostic; inspired by cited methods, not a published benchmark item.'
        assert len(c['prompt'])<3000
    return cases


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'cases.jsonl')
    a=p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite existing cases. Choose another --output.')
    a.output.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in make_suite()),encoding='utf-8')
    print('Wrote 120 cases: 30 logic originals + 60 variants + 20 critiques + 10 question-generation cases.')
