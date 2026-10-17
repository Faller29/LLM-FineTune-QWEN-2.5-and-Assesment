ENNIS-WEIR ADD-ON FOR YOUR EXISTING TSA PROJECT

Extract the ZIP, then copy its EW_Evaluation folder into the same project root
as TSA_Evaluation. Avoid an extra Ennis_Weir_Add_On folder level in that root.
Expected folders:
  D:\MSCS213-Simulation - Copy\TSA_Evaluation\
  D:\MSCS213-Simulation - Copy\EW_Evaluation\
  D:\MSCS213-Simulation - Copy\results\qwen_lora\best\

Start with COMMANDS_WINDOWS_CMD.txt. Run each stage yourself in CMD.
Use your existing .venv and trained LoRA adapter. No retraining is required.
Source import, review scoring, and software tests use only Python's standard
library. Inference reuses the TSA loader and its installed Torch/Transformers/PEFT.

WHAT THIS ADDS
The authors' original Moorburg letter plus all three added scenarios:
  EW-ORIGINAL: 9 scoring units, maximum 29.
  EW-NEW-1: 5 scoring units in 3 response paragraphs, maximum 15.
  EW-NEW-2: 4 scoring units in 3 response paragraphs, maximum 12.
  EW-NEW-3: 3 scoring units in 3 response paragraphs, maximum 9.
Total: 21 scoring units, nominal maximum 65 points per complete round.
Default run: four complete prompts, one greedy response per scenario.

The published instructions are imported without rewriting. Each complete
scenario reaches the model in a fresh chat; reference answers are never sent.
The same 'detailed thinking on' system instruction used by your TSA run remains
the default. It is an instruction, not evidence of internal process faithfulness.

DOWNLOAD / PREPARE
prepare_ew.py imports all_items.xlsx and Scoring Manual.docx from a pinned
commit of the authors' repository. It verifies SHA256 fingerprints. If your
TSA_Evaluation\sources\all_items.xlsx is the same edition, it reuses that file.
It normally downloads only the small scoring manual. A changed source is
rejected instead of silently changing the test.

Repository:
https://github.com/Jiang-Liming/Evaluating-Critical-Thinking-of-LLM
Pinned commit: 595874976e96861c2541d841d1375a23237e92e7
Workbook:
https://raw.githubusercontent.com/Jiang-Liming/Evaluating-Critical-Thinking-of-LLM/595874976e96861c2541d841d1375a23237e92e7/all_items.xlsx
Manual:
https://raw.githubusercontent.com/Jiang-Liming/Evaluating-Critical-Thinking-of-LLM/595874976e96861c2541d841d1375a23237e92e7/Scoring%20Manual.docx

Offline fallback after downloading those exact files yourself:
python EW_Evaluation\prepare_ew.py --workbook "C:\Users\YOUR_USERNAME\Downloads\all_items.xlsx" --manual "C:\Users\YOUR_USERNAME\Downloads\Scoring Manual.docx"

The distribution contains importer code and fingerprints. Published questions
and full reference answers are obtained from their source during preparation.
Generated PROMPTS_FOR_INSPECTION.txt contains test inputs only.
Generated REFERENCE_ANSWERS_DO_NOT_PROMPT.json is for graders; do not paste it
into Qwen or include it in fine-tuning before the assessment.

RESULTS
results\ew_final\Compile_Answer.zip contains exact prompts/responses,
model/adapter identity, source references, and the review template.
Upload that ZIP to this conversation for an AI-assisted rubric review.
Before review, the score is PENDING. This is intentional: the script does not
use keywords or an answer-letter parser to claim that an essay is correct.

After a completed ratings JSON is imported, RESULTS.txt and summary.json show
the released-manual score, the separately labeled article-wording sensitivity
score, and reviewer identity. Text quotations bind scoring judgments to the
actual raw responses. Multiple independent review files can be supplied together
for descriptive averaging. Do not label an AI-assisted review as a human rating.

If --tsa-out results\tsa_final is supplied, the runner verifies the same base
model and adapter fingerprints and includes the existing TSA answer score in
COMBINED_ASSESSMENT.json. The TSA and EW scores remain separate.

WHAT 'COMPLETE' MEANS HERE
This adds the full released EW content to your existing 35-original-item TSA
evaluation. It completes both instrument components in this CPU-adapted project.
It does not reproduce the entire study: 70 TSA items, multiple commercial models,
194 students, 150 rounds/model, three trained human raters, and rating-group
equating are outside the default project protocol. METHOD_NOTES.txt lists these
differences and discrepancies within the released scoring documentation.

SOURCE
Jiang, L., Cui, Y., Liu, R., & Luo, F. (2025). Evaluating the Critical Thinking of
Large Language Models: Insights and Limitations. Journal of Pacific Rim Psychology.
https://doi.org/10.1177/18344909251406111

The original instrument is Ennis & Weir (1985). This add-on follows the Jiang
release for references and nominal item maxima; it does not substitute the
original manual's -1-to-3 judgment scale for Jiang's base-credits-plus-deductions.
