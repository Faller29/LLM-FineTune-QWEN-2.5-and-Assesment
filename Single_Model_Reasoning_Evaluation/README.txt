SINGLE-MODEL REASONING AND CRITICAL-REASONING EVALUATION
Target: your fine-tuned Qwen2.5 plus its trained LoRA adapter
Date: 13 October 2026

WHAT THIS PACKAGE DOES

Calls your existing local web_app.py server with choice='tuned', mode='on'
by default, task='chat', and empty history. The base model is never evaluated
as a separate condition. Only the tuned response is requested and scored.
The program uses Python's standard library; no additional pip installs or paid
grader API are required. No existing project files or training data are modified.

RECOMMENDED METHOD

Use a small paper-informed diagnostic with known-answer automated scoring and
human review of open-ended critical questions. This is a practical first study,
not a reproduction of published benchmark scores or a global thinking certificate.

Packaged cases: 120 original custom cases.
Development reserve: 12 cases (three logic families with variants, two critiques,
one critical-question argument). The smoke command runs six of these, one per
original question/critique/argument. The final command excludes ALL reserve cases.
Default final run: 108 model generations:
  27 original logic questions, balanced 9 TRUE / 9 FALSE / 9 UNKNOWN;
  54 related variants (27 paraphrases and 27 altered queries/facts);
  18 critique cases (9 correct proposed classifications and 9 incorrect);
  9 arguments, each requesting three critical questions (27 requested slots).

CRITERIA AND RESEARCH BASIS

1. Supported conclusions and insufficient information:
   ProofWriter: https://doi.org/10.18653/v1/2021.findings-acl.317
   FOLIO: https://doi.org/10.18653/v1/2024.emnlp-main.1229
   This adaptation uses simple grounded forward rules with explicit negation.
   It does not implement the full FOLIO first-order logic task or train ProofWriter.
   TRUE: query follows. FALSE: explicit negation follows. UNKNOWN: neither follows.
   A small closure engine verifies the intended formal gold labels, independently
   of the LLM's answers. Human checking of natural-language wording is still needed.

2. Error detection and correction:
   CriticBench: https://doi.org/10.18653/v1/2024.findings-acl.91
   Ask the model to judge a supplied truth classification, identify an error and
   give a corrected classification. Include correct proposals so that criticizing
   everything fails. This is a narrower adaptation of critique/correction, not a
   general assessment of open-ended argument quality or self-correction.

3. Useful critical questions:
   CQs-Gen: https://doi.org/10.18653/v1/2025.findings-emnlp.302
   Request exactly three questions that challenge an argument's assumptions or
   evidence. Review the resulting nine responses in the generated HTML page.
   Judge meaning using useful/unhelpful/invalid categories. Missing question slots
   remain in the denominator. Score only the first three and give duplicates no
   additional useful credit; these missing/duplicate conventions are adaptations.
   Suggested reference questions are examples, not mandatory wording or exhaustive
   gold answers. Other specific, valid challenges can receive useful credit.

4. Robustness:
   HANS: https://doi.org/10.18653/v1/P19-1334
   Argument shortcuts: https://doi.org/10.18653/v1/P19-1459
   Paraphrases preserve answers. Altered questions/facts change the correct label.
   Report originals and variants separately; do not treat 81 logic responses as
   81 independent questions. This adaptation tests narrow behavioral consistency.

5. Scoring tools/documentation:
   https://inspect.aisi.org.uk/scorers.html
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
   These documents support separating response scoring from aggregate metrics.
   This lightweight client does NOT require either Inspect or scikit-learn.

WINDOWS CMD: STEP BY STEP

1. Extract this folder into D:\MSCS213-Simulation so you have:
   D:\MSCS213-Simulation\Single_Model_Reasoning_Evaluation\evaluate.py

2. CMD window 1: use your EXISTING environment and actual trained adapter.

cd /d D:\MSCS213-Simulation
.venv\Scripts\activate.bat
python web_app.py --adapter results\qwen_lora\best --max-input 512

Leave that window open. If your adapter is elsewhere, change only its path.

3. CMD window 2:

cd /d D:\MSCS213-Simulation
.venv\Scripts\activate.bat
python Single_Model_Reasoning_Evaluation\evaluate.py validate

Stop if validation fails. Have another person check the intended wording and
reference answers before drawing conclusions from the final scores.

4. Small development check (six generations):

python Single_Model_Reasoning_Evaluation\evaluate.py run --smoke --out results\single_model_smoke
python Single_Model_Reasoning_Evaluation\evaluate.py summarize --out results\single_model_smoke
type results\single_model_smoke\summary.json

Inspect responses.jsonl for readable, complete answers and requested markers.
The output budget defaults to 384 tokens, the current server's maximum.
The program defaults to detailed thinking on. This is a system instruction;
it is not proof of a separate internal reasoning mechanism. Fix the chosen mode,
prompt and budget before final evaluation. A large number of incomplete answers
means you cannot attribute low scores solely to reasoning inability.

5. Default final diagnostic (108 generations):

python Single_Model_Reasoning_Evaluation\evaluate.py run --out results\single_model_final
python Single_Model_Reasoning_Evaluation\evaluate.py summarize --out results\single_model_final
type results\single_model_final\summary.json

Generation can be slow on CPU. Estimate time from the six-case development run.
The original server loads one model; this client does not load a second copy.
Input that exceeds the server's token limit causes an error, not silent truncation.
If interrupted, wait for any in-flight request to finish, or restart the server,
then resume with unchanged adapter, suite, client and settings:

python Single_Model_Reasoning_Evaluation\evaluate.py run --out results\single_model_final --resume

6. Review critical questions:

start "" results\single_model_final\critical_question_review.html

Select useful, unhelpful, invalid or missing for all three requested slots in each
argument. Click Download ratings.json. Move the downloaded file into
results\single_model_final, then run:

python Single_Model_Reasoning_Evaluation\evaluate.py summarize --out results\single_model_final --ratings results\single_model_final\ratings.json

This creates summary_with_human_ratings.json. Human scores are explicitly marked
as human ratings; they are not automatically verified facts. For a formal paper,
use two independent reviewers, retain separate ratings files and report agreement
before adjudication. The automatic package does not compute interrater agreement.

7. Audit explanation validity separately:

start "" results\single_model_final\explanation_audit.txt

The file preserves every automatically scored response and its reference. Review
all originals if feasible, or predefine a random sample. Record whether each
explanation uses the supplied facts and valid forward inference. Do not treat a
correct final label as evidence that its explanation is correct. These audit
scores are not computed automatically; report them as manual measurements.

OUTPUTS AND INTERPRETATION

run_config.json: model identity, adapter fingerprints, suite/client hashes,
selected case IDs, mode and budget. Record base-model revision, tokenizer revision,
package versions, hardware and thread count separately. Do not change cached base
weights, tokenizer, server code or decoding while the evaluation is in progress.

responses.jsonl: exact prompt, raw response, timing, token counts, reference label
and parsed decision. It contains the tuned model only.

summary.json:
- completion: complete or incomplete, with missing IDs;
- logic.original: answer accuracy, confusion matrix and classification macro-F1;
- logic.paraphrase / changed: separate variant performance;
- logic_depth_originals: accuracy for one-, two- and three-step supported cases,
  plus unknown cases (depth=null because no supporting chain exists);
- robustness: families where all three responses are correct;
- critique: precision, recall, F1 with INCORRECT as the positive class, verdict
  accuracy, false alarms and corrected-classification accuracy;
- critical_questions: responses awaiting human review;
- possible_budget_limit: heuristic flag, based on re-tokenized response length;
- seconds: recorded inference time, not a predicted runtime guarantee.

Precision = genuine detected errors / all claimed errors.
Recall = detected errors / all actual errors.
F1 = 2 TP / (2 TP + FP + FN).
Logic macro-F1 averages the F1 of TRUE, FALSE and UNKNOWN. This is CLASSIFICATION
F1, not token-overlap F1. Undefined precision/recall values are set to zero and
class support counts are retained. No zero score is used for an unmeasured rubric.

CORRECTION ACCURACY means a corrected truth label on the originally incorrect
proposals. It does not automatically grade the wording of a repair or the validity
of its explanation. Detection-and-correction success requires both judgments.

Strict formatting is requested: FINAL: TRUE/FALSE/UNKNOWN for logic; VERDICT:
CORRECT/INCORRECT and CORRECTED: TRUE/FALSE/UNKNOWN for critique. Missing markers
count as unsuccessful answers in strict accuracy. Audit raw responses separately
to distinguish format-only failures. Never silently drop these failures.

SCOPE OF CLAIMS

This is an introductory, custom diagnostic covering a narrow set of behaviors.
It can support statements such as 'the model correctly classified X of 27 new
forward-rule questions and detected Y of 9 incorrect proposed classifications'.
It does not establish human-level critical thinking, arbitrary first-order logic,
internal chain-of-thought faithfulness, or official benchmark performance.

There is no paper-backed universal pass threshold in this package. Report actual
counts and performance by criterion rather than inventing an 80% certification.
Wilson intervals are descriptive under independent-case sampling assumptions;
purposively authored questions and correlated variants limit broad inference.
Public benchmark testing and larger independently authored sets would strengthen
a final research claim. Do not tune the model or choose its checkpoint on these
final questions; use the reserved development set for implementation decisions.

VERIFICATION

The suite's formal labels, parser, metrics with known TP/FP/FN/TN counts, request
shape, tuned-only behavior, resume guards and human-review result import were
tested offline with a mock server. The mock server's scores are NOT Qwen scores.
The actual pretrained weights and trained adapter were not uploaded, so real
model inference has not been run in this environment.

Optional offline verification on your PC:
python Single_Model_Reasoning_Evaluation\test_evaluate.py

The build_suite.py file lets you inspect exactly how the diagnostic was created.
It refuses to overwrite cases.jsonl. Do not regenerate/edit the frozen suite
mid-run. To generate an inspectable copy, supply --output with a new filename.
