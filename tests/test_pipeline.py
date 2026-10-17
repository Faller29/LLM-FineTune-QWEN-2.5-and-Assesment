import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import (boxed_final,boxed_value,empty_thinking,nano_membership,
                    test_overlap,training_tokens,training_prompt, read_jsonl)


def load_script(name):
    spec=importlib.util.spec_from_file_location('module_'+name,ROOT/'scripts'/name)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

class FakeTokenizer:
    name_or_path='dummy-qwen'
    eos_token='[END]'
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return '\n'.join(m['role']+':'+m['content'] for m in messages)+'\nassistant:'
    def __call__(self,text,add_special_tokens=False):
        return {'input_ids':[i for i,_ in enumerate(text.split())]}

class FakeLanguage:
    def predict(self,text,k=1):
        label='__label__xx' if 'espagnol' in text else '__label__en'
        return ([label],[.95])


def raw(q,mode='off',category='chat',answer='A straightforward response',used=None):
    if used is None:used=['Nano','Ultra']
    return {'input':[{'role':'user','content':q}], 'output':answer,
            'reasoning':mode,'category':category,'used_in_training':used,
            'system_prompt':'detailed thinking '+mode}

class TestNvidiaRecipe(unittest.TestCase):
    def test_filter_semantics(self):
        self.assertTrue(nano_membership({'used_in_training':['Ultra','Nano']}))
        self.assertFalse(nano_membership({'used_in_training':['Ultra']}))
        self.assertTrue(empty_thinking('<think> \n </think>ok'))
        self.assertFalse(empty_thinking('<think>do math</think>ok'))
        self.assertFalse(boxed_final('<think>\\boxed{5}</think> result 5','on'))
        self.assertTrue(boxed_final('<think>reason</think>\\boxed{5}','on'))
        self.assertEqual(boxed_value('x \\boxed{\\frac{1}{2}}'),'\\frac{1}{2}')
        t=FakeTokenizer()
        result=training_prompt(t,[{'role':'user','content':'Hi'}],'on')
        self.assertIn('detailed thinking on',result)
        ids,labels=training_tokens(t,{'input':result,'output':'Hey'})
        self.assertEqual(len(ids),len(labels))
        self.assertIn(-100,labels)
        self.assertNotEqual(labels[-1],-100)

    def test_filtered_split_and_budget(self):
        mod=load_script('02_curate_nvidia.py')
        chooser=load_script('03_cpu_training_selection.py')
        with tempfile.TemporaryDirectory() as d:
            math_file=Path(d)/'math.jsonl'
            chat_file=Path(d)/'chat.jsonl'
            mr=[raw(f'Solve English math question number {i}',mode='on',category='math',
                    answer=f'<think>multiply small numbers carefully number {i}</think>\\boxed{{{i}}}') for i in range(30)]
            mr+= [raw('bad math answer',mode='on',category='math',answer='<think>yes</think>No box'),
                  raw('empty thought',mode='on',category='math',answer='<think> </think>\\boxed{1}'),
                  raw('non nano',mode='on',category='math',answer='<think>ok</think>\\boxed{1}',used=['Super'])]
            cr=[raw(f'An English chat question number {i}',mode='off') for i in range(30)]
            cr.append(raw('espagnol no thanks',mode='off'))
            for path,items in [(math_file,mr),(chat_file,cr)]:
                with path.open('w',encoding='utf-8') as out:
                    for obj in items:out.write(json.dumps(obj)+'\n')
            splits,report=mod.curate([(math_file,'math'),(chat_file,'chat')],
                        FakeTokenizer(),FakeLanguage(),8192,16384,0,0,42,.1,.1)
            self.assertEqual(report['curated_total'],60)
            self.assertEqual(report['filter_counts']['reject_unboxed_math_final'],1)
            self.assertEqual(report['filter_counts']['reject_empty_think'],1)
            self.assertEqual(report['filter_counts']['reject_non_nano'],1)
            self.assertEqual(report['filter_counts']['reject_non_english_fasttext'],1)
            self.assertFalse(test_overlap(list(splits.values())))
            self.assertEqual(sum(len(v) for v in splits.values()),60)
            self.assertGreater(len(splits['test']),0)
            self.assertIn('math/on',report['composition']['train'])
            self.assertIn('chat/off',report['composition']['train'])
            # A hardware budget can exclude rows WITHOUT changing curated data.
            limited,counts=chooser.select(splits['train'],10,0)
            self.assertLess(len(limited),len(splits['train']))
            self.assertEqual(counts['selected']+counts['not_trainable_with_cpu_context'],len(splits['train']))
            self.assertEqual(len(splits['train']),report['train'])

if __name__=='__main__':unittest.main()
