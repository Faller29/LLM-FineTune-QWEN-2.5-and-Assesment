"""Reuse the existing TSA model loader; add temperature control and progress."""
from __future__ import annotations
import importlib
import sys
import time
from pathlib import Path


def existing_tsa_modules():
    folder = Path(__file__).resolve().parent.parent / "TSA_Evaluation"
    if not (folder / "evaluate_tsa.py").is_file() or not (folder / "tsa_core.py").is_file():
        raise ValueError("Extract EW_Evaluation beside your existing TSA_Evaluation folder.")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return importlib.import_module("evaluate_tsa"), importlib.import_module("tsa_core")


def backend_factory(args):
    tsa, _ = existing_tsa_modules()

    class Backend(tsa.DirectBackend):
        def describe(self):
            identity = super().describe()
            identity["do_sample"] = "per_completion: false at T=0, true at T>0"
            return identity

        def generate_task(self, prompt, task):
            import torch
            from transformers import StoppingCriteria, StoppingCriteriaList
            if self.model is None:
                from peft import PeftModel
                from transformers import AutoModelForCausalLM
                print("Loading the same cached Qwen base and trained LoRA adapter used by TSA...", flush=True)
                base = AutoModelForCausalLM.from_pretrained(
                    self.model_name, local_files_only=not self.args.allow_model_download,
                    torch_dtype=torch.float32, use_safetensors=True,
                ).to("cpu")
                self.model = PeftModel.from_pretrained(base, str(self.adapter)).eval()
            torch.manual_seed(task["generation_seed"])
            full = self.chat_input(prompt)
            encoded = self.tokenizer(full, return_tensors="pt", add_special_tokens=False)
            input_tokens = int(encoded["input_ids"].shape[1])
            if input_tokens > self.args.max_input:
                raise ValueError("Full scenario exceeds the input limit; no truncation was performed.")
            started = time.monotonic()

            class Heartbeat(StoppingCriteria):
                def __init__(self):
                    self.last = started

                def __call__(self, input_ids, scores, **kwargs):
                    now = time.monotonic()
                    if now - self.last >= 30:
                        print(f"  generated {input_ids.shape[1] - input_tokens} tokens; "
                              f"{now - started:.0f}s elapsed", flush=True)
                        self.last = now
                    return False

            kwargs = {"do_sample": task["temperature"] > 0,
                      "max_new_tokens": self.args.max_new_tokens,
                      "pad_token_id": self.tokenizer.eos_token_id,
                      "stopping_criteria": StoppingCriteriaList([Heartbeat()])}
            if task["temperature"] > 0:
                kwargs.update(temperature=task["temperature"], top_p=1.0, top_k=0)
            with torch.inference_mode():
                generated = self.model.generate(**encoded, **kwargs)
            output = generated[0][input_tokens:]
            count = int(output.numel())
            eos = self.model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
            finish = "eos" if count and int(output[-1]) in eos_ids else (
                "length" if count >= self.args.max_new_tokens else "other")
            return {"response": self.tokenizer.decode(output, skip_special_tokens=True),
                    "input_tokens": input_tokens, "output_tokens": count,
                    "finish_reason": finish, "model_input": full}

    return Backend(args)
