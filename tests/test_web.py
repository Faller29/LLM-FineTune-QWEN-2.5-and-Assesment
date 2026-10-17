import importlib.util
from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import sys
import threading
import unittest
from urllib.request import urlopen,Request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import web_app as ui

class WebTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ui.setup()
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),ui.LabHandler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.root=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
    def test_routes(self):
        for path in ['/','/static/app.js','/static/style.css','/api/status','/api/examples','/api/experiment','/api/loss']:
            with urlopen(self.root+path,timeout=4) as reply:
                self.assertEqual(reply.status,200)
        with urlopen(self.root+'/api/status',timeout=4) as reply:
            status=json.load(reply)
            self.assertEqual(status['model'],'Qwen/Qwen2.5-0.5B-Instruct')
    def test_generate_http_contract_without_loading_model(self):
        original=ui.run_inference
        ui.run_inference=lambda body:({'prompt':'mock','mode':body['mode'],
            'outputs':{'base':{'text':'mock','seconds':0,'output_tokens':1,'boxed_answer':None}}},200)
        try:
            request=Request(self.root+'/api/generate',data=json.dumps({'question':'1+2','mode':'on'}).encode(),
                           headers={'Content-Type':'application/json'},method='POST')
            with urlopen(request,timeout=4) as reply:
                self.assertEqual(json.load(reply)['outputs']['base']['text'],'mock')
        finally:
            ui.run_inference=original

if __name__=='__main__':unittest.main()
