"""Download and smoke-test Qwen2.5-0.5B before the expensive data processing."""
import argparse
from common import BASE_MODEL,generate,training_prompt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',default=BASE_MODEL)
p.add_argument('--threads',type=int,default=4)
p.add_argument('--max-new-tokens',type=int,default=96)
a=p.parse_args()
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
torch.set_num_threads(a.threads)
tok=AutoTokenizer.from_pretrained(a.model)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32).eval()
for mode in ('off','on'):
    prompt=training_prompt(tok,[{'role':'user','content':'Find the units digit of 13 times 41.'}],mode)
    output=generate(model,tok,prompt,a.max_new_tokens)
    print(f'[{mode}] INPUT TOKENS:',len(tok(prompt,add_special_tokens=False)['input_ids']))
    print(f'[{mode}] RESPONSE:',repr(output))
print('Model loading and CPU inference completed. This does not demonstrate new reasoning ability.')
