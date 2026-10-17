"""Offline checks only: no real model scores are produced."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import evaluate as e
from build_suite import make_suite,solve


class Checks(unittest.TestCase):
    def test_forward_logic_and_nonreversible_rules(self):
        p=('A','calm',True)
        q=('A','bright',True)
        self.assertEqual(solve([p],[(p,q)],q),'TRUE')
        self.assertEqual(solve([q],[(p,q)],p),'UNKNOWN')
        self.assertEqual(solve([p],[(p,('A','bright',False))],q),'FALSE')
        with self.assertRaises(ValueError):
            solve([q,('A','bright',False)],[],q)

    def test_suite_and_parser(self):
        suite=make_suite()
        e.validate(suite)
        self.assertEqual(len(suite),120)
        originals=[x for x in suite if x['task']=='logic' and x['variant']=='original']
        self.assertEqual([sum(x['gold']==g for x in originals) for g in e.LABELS],[10,10,10])
        crit=[x for x in suite if x['task']=='critique']
        self.assertEqual(sum(x['gold_verdict']=='INCORRECT' for x in crit),10)
        self.assertIsNone(e.extract('I think TRUE.','FINAL',e.LABELS))
        self.assertIsNone(e.extract('<think>FINAL: TRUE','FINAL',e.LABELS))
        self.assertEqual(e.extract('<think>FINAL: FALSE</think>\nFINAL: TRUE','FINAL',e.LABELS),'TRUE')
        self.assertEqual(e.extract('VERDICT: INCORRECT\nCORRECTED: UNKNOWN','VERDICT',['CORRECT','INCORRECT']),'INCORRECT')

    def test_known_f1_counts_and_format_failures(self):
        def row(gold,pred):
            return dict(gold='TRUE',gold_verdict=gold,output=dict(verdict=pred,parsed='TRUE'))
        rows=[row('INCORRECT','INCORRECT') for _ in range(15)]+[row('INCORRECT','CORRECT') for _ in range(5)]+[row('CORRECT','INCORRECT') for _ in range(3)]+[row('CORRECT','CORRECT') for _ in range(17)]
        s=e.critique_scores(rows)
        self.assertEqual((s['tp'],s['fp'],s['fn'],s['tn']),(15,3,5,17))
        self.assertAlmostEqual(s['error_f1'],30/38)
        self.assertEqual(s['verdict_accuracy'],.8)
        classes=e.classification([dict(gold='TRUE',output=dict(parsed='TRUE')),dict(gold='TRUE',output=dict(parsed=None)),dict(gold='FALSE',output=dict(parsed='UNKNOWN'))])
        self.assertEqual(classes['accuracy'],1/3)
        self.assertEqual(classes['confusion_matrix']['TRUE']['INVALID'],1)

    def test_only_tuned_request_resume_and_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            adapter=root/'adapter'
            adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            lookup={x['prompt']:x for x in make_suite()}
            received=[]
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*args):
                    pass
                def respond(self,data):
                    b=json.dumps(data).encode()
                    self.send_response(200)
                    self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                def do_GET(self):
                    self.respond(dict(adapter_available=True,adapter=str(adapter),model='MOCK-NOT-QWEN',max_input=512))
                def do_POST(self):
                    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    received.append(body)
                    x=lookup[body['question']]
                    text='MOCK ONLY\n'+('FINAL: '+x['gold'] if x['task']=='logic' else
                        'VERDICT: '+x['gold_verdict']+'\nCORRECTED: '+x['gold'] if x['task']=='critique' else
                        '1. What evidence supports it?\n2. Was it tested?\n3. Are there alternatives?')
                    self.respond(dict(prompt=body['question'],input_tokens=100,outputs={'tuned':dict(text=text,seconds=.01,output_tokens=20)}))
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            out=root/'out'
            args=[sys.executable,str(e.ROOT/'evaluate.py'),'run','--smoke','--url',f'http://127.0.0.1:{server.server_port}','--out',str(out)]
            try:
                p=subprocess.run(args,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
                self.assertEqual(len(received),6)
                self.assertTrue(all(x['choice']=='tuned' and x['task']=='chat' and x['history']==[] for x in received))
                p=subprocess.run(args+['--resume'],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
                self.assertEqual(len(received),6)
                p=subprocess.run([sys.executable,str(e.ROOT/'evaluate.py'),'summarize','--out',str(out)],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
                report=json.loads((out/'summary.json').read_text())
                self.assertTrue(report['completion']['complete'])
                self.assertEqual(report['logic']['original']['accuracy'],1)
                self.assertEqual(report['critique']['error_f1'],1)
                self.assertNotIn('FINAL: TRUE',(out/'critical_question_review.html').read_text())
                ratings=root/'ratings.json'
                ratings.write_text(json.dumps([dict(id='Q10',ratings=['useful','unhelpful','missing'])]))
                p=subprocess.run([sys.executable,str(e.ROOT/'evaluate.py'),'summarize','--out',str(out),'--ratings',str(ratings)],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
                report=json.loads((out/'summary_with_human_ratings.json').read_text())
                self.assertEqual(report['critical_questions']['useful_fraction'],1/3)
                (adapter/'adapter_config.json').write_text('{"changed":true}')
                p=subprocess.run(args+['--resume'],capture_output=True,text=True)
                self.assertNotEqual(p.returncode,0)
                self.assertIn('Cannot resume',p.stderr)
            finally:
                server.shutdown()
                server.server_close()


if __name__=='__main__':
    unittest.main()
