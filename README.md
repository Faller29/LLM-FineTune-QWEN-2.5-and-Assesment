# NVIDIA-aligned reasoning SFT lab / CPU, Windows 11

**Source:** NVIDIA, *Train a Reasoning-Capable LLM in One Weekend with NVIDIA NeMo* (22 July 2025):
https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/

**Scope:** An independent CPU implementation of the **published filtering methodology**. The curation rules match the **blog's stated rules**; this does NOT use the literal NeMo Curator code or NVIDIA's NeMo Framework trainer. CPU SFT still requires a separate hardware-constrained sample, checkpoint and batch-size choices. There is no claim of equivalent reasoning performance.

## Why this is a clean rebuild

- Base model is **`Qwen/Qwen2.5-0.5B-Instruct`** (0.49B, decoder-only, native chat template), not the earlier encoder-decoder FLAN-T5 model or NVIDIA's Llama 3.1 8B.
- Curation retains NVIDIA's **Nano, math_v1.1 + chat, FastText English, boxed math, nonempty thinking, 8192 input/16384 completion, system toggles, chat template, length curriculum**.
- Unlike earlier labs, **no 256- or 384-token restriction, no ASCII language heuristic, no generated math-off pairs, no mutation/truncation of NVIDIA targets in curation**.
- Because a 16 GB CPU cannot practically train 16k-token continuations, Stage 3 selects a **separate CPU training subset** from the faithfully curated corpus. Any skipped records stay in `data/curated` and are counted.
- Training and evaluation are still real PyTorch LoRA operations. No fabricated scores are output.

## Equipment and practical limits

Target Windows 11, Ryzen 5 3400G, 16 GB RAM. RX 560 **not used** for training. On CPU, even 0.49B with LoRA can run slowly; test a few steps before running tens or hundreds. Qwen has a 32K architecture context limit, but CPU memory prevents you from training at that limit.

Your **100 GB available storage** can likely fit a **math_v1.1 (~41.4 GB) + chat (~255 MB)** selective download. Do NOT clone the complete NVIDIA dataset (~130 GB) onto that drive. Allow additional space for temporary files, cache, output and Windows free space. Downloads require internet. Avoid downloading 41 GB before testing the environment and small model.

## Stage 0. Windows virtual environment (PowerShell)

```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -r requirements-curation.txt
python scripts/00_verify_environment.py
python scripts/00_test_model.py
```

For `WinError 126`, repair Microsoft's Visual C++ x64 runtime, then verify `python -c "import torch; print(torch.__version__)"` before proceeding. NVIDIA CUDA and `bitsandbytes` are not required. The first model test downloads the pretrained Qwen weights from Hugging Face.

## Stage 1. NVIDIA dataset download (PowerShell terminal)

Original NVIDIA tutorial uses `git lfs install` followed by cloning *the entire ~130 GB dataset*. Since you have about 100 GB free, select **only the two files specified by NVIDIA's curation**. `hf download` is Hugging Face's official command-line downloader:

```
python -m pip install --upgrade huggingface_hub
hf download nvidia/Llama-Nemotron-Post-Training-Dataset SFT/chat/chat.jsonl --repo-type dataset --local-dir data/source
hf download nvidia/Llama-Nemotron-Post-Training-Dataset SFT/math/math_v1.1.jsonl --repo-type dataset --local-dir data/source
```

The math file is ~41.4 GB, and will take time. This is the original NVIDIA source, not a fabricated dataset and not a new Nano-specific dataset. The Nano subset is identified by `used_in_training` **during curation**.

Download the **exact FastText model linked in NVIDIA's tutorial**:

```
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz" -OutFile "data/lid.176.ftz"
```

Inspect raw files (only scans the first 1000 records of each for a quick read):

```
python scripts/01_inspect_raw.py --limit-lines 1000 --show 2
```

## Stage 2. NVIDIA-style curation

The blog's published command uses `--filename-filter chat math_v1.1`, `--max-token-count 8192`, `--max-completion-token-count 16384`, FastText English identification and a tokenizer adapted to the model. Here is our corresponding independent implementation:

```
python scripts/02_curate_nvidia.py --math data/source/SFT/math/math_v1.1.jsonl --chat data/source/SFT/chat/chat.jsonl --lang-model data/lid.176.ftz --model Qwen/Qwen2.5-0.5B-Instruct --max-token-count 8192 --max-completion-token-count 16384 --max-per-source 1500
```

**Important:** `--max-per-source 1500` is an **explicit sampling ceiling**, not one of NVIDIA's filtering rules. It stops at the first 1500 qualifying rows from EACH source to make the lab practical. If you want *all* qualifying source rows, use `--max-per-source 0`, but processing the large math file can be very slow and the resulting curated dataset may be very large. The default does not pretend to give a random or representative sample.

To run a quick dry run without scanning the whole file, use `--max-lines 2000` (explicitly limits rows scanned per source, not a quality filter).

Outputs: `data/curated/train.jsonl`, `val.jsonl`, `test.jsonl`, and `curation_report.json`. Train/validation/test grouping is an extra methodological requirement so the evaluation has no repeated question across splits. The model's **Qwen tokenizer/chat template** replaces Llama's tokenizer and chat template (required by changing the base model). Sources and answers remain NVIDIA's.

Inspect curation statistics:

```
Get-Content data/curated/curation_report.json
```

## Stage 3. CPU training eligibility: NOT part of NVIDIA's filter

```
python scripts/03_cpu_training_selection.py --sequence-budget 512 --train-limit 400 --val-limit 32 --test-limit 32
Get-Content data/cpu/cpu_selection_report.json
```

This stage only SELECTS full records that fit the CPU budget; it does not shorten or rewrite them. The original NVIDIA-filtered corpus is preserved. If you have too few math/on or chat/off examples, increase the curation sampling ceiling and rerun curation rather than inventing replacement answers. On 16GB, try `--sequence-budget 768` only after the 512-token experiment runs reliably.

## Stage 4. Evaluate the original model

```
python scripts/04_baseline.py --limit 8 --max-new-tokens 128
```

Output: `results/baseline.jsonl`. The same held-out set is used later in Stage 6; no training examples are included.

## Stage 5. LoRA SFT on CPU

Very small stability/compatibility run (START HERE):

```
python scripts/05_train_lora.py --max-steps 8 --grad-accumulation 2 --rank 8 --alpha 16 --max-sequence 512 --output results/qwen_smoke
```

If it completes and memory remains acceptable, a more meaningful test:

```
python scripts/05_train_lora.py --max-steps 80 --grad-accumulation 4 --rank 8 --alpha 16 --max-sequence 512 --eval-every 10 --output results/qwen_lora
```

NVIDIA's original used LoRA rank 64/alpha128, LR1e-4, cosine schedule with 5% warmup, weight decay 0.001, effective batch256, >=2000 steps. This lab preserves LR/scheduler/warmup/decay and the LoRA method, while reducing rank, effective batch, context and steps because of CPU/RAM limitations. Increase those only after measuring CPU performance.

Checkpoint outputs: `results/qwen_lora/best`, `results/qwen_lora/final`, `results/qwen_lora/loss.csv`, and `training_config.json`.

## Stage 6. Held-out comparison

```
python scripts/06_compare.py --adapter results/qwen_lora/best --limit 8 --max-new-tokens 128
```

Outputs: `results/compare_predictions.jsonl`, `results/compare_metrics.json`. Scoring uses strict boxed answer matching and thinking-format checks, **not GPQA/MMLU equivalence**. Larger held-out tests are better if you can afford the CPU generation time.

## Stage 7. Local chat frontend

```
python web_app.py --adapter results/qwen_lora/best --max-input 512
```

Go to `http://127.0.0.1:7860/` in your browser. It uses the **same Qwen2.5 chat template and `detailed thinking on/off`** controls as training, and can show base and LoRA responses side-by-side. Before training, start the server and choose **Base model** in the dropdown. The Python standard library runs the web server; no cloud inference, Node or external app server is required. The first prompt loads the model and may take time on a CPU.

## Limitations and honest report wording

> This study adapts NVIDIA's published dataset selection and filtering methodology using the NVIDIA Llama Nemotron Post-Training Dataset. It replaces Llama 3.1 8B with the smaller Qwen2.5-0.5B-Instruct model and NeMo Framework with CPU-based PyTorch/PEFT LoRA. A separate documented training-stage context limit makes execution feasible on 16 GB RAM. The experiment measures model-specific behavior, not a reproduction of NVIDIA's benchmark gains.

Specific differences: original NVIDIA code uses NeMo Curator/Framework and NVIDIA's tokenizer; our curation is a Python implementation of the **blog-described rules**, and our CPU learner is a separate implementation. Different tokenization may change which length-filtered records pass. Dataset composition may be biased by the explicit sample ceiling. NVIDIA's 8B quality/benchmark results cannot be extrapolated to the 0.49B model.

For your academic record save screenshots of: raw examples; FastText+Nano filter report; curation counts; CPU eligibility report; base predictions; LoRA optimizer steps and validation; same-question base vs LoRA comparisons.
#   L L M - F i n e T u n e - Q W E N - 2 . 5 - a n d - A s s e s m e n t 
 
 