TSA EVALUATION FOR YOUR FINE-TUNED QWEN2.5
Prepared 13 October 2026

This add-on runs your existing trained Qwen model on 35 published TSA questions.
It scores the answers against UCLES answer keys and saves the complete responses
so you can send them to ChatGPT for a separate review of the explanations.

It uses your existing Python environment, PyTorch, Transformers and PEFT.
The importer and scorer use only Python's standard library. No paid grading
service, Excel installation, API key or additional Python package is required.
No training is performed. Existing training data, adapters and scripts are not
changed. Run the new files from your project root after extracting this folder.

WHAT IS THE TEST?

The question selection comes from the TSA_original worksheet in the materials
for Jiang et al. (2025), Evaluating the Critical Thinking of Large Language
Models: Insights and Limitations.
DOI: https://doi.org/10.1177/18344909251406111
Materials: https://github.com/Jiang-Liming/Evaluating-Critical-Thinking-of-LLM

The workbook supplies the criterion labels and question texts. It does not
supply TSA answer keys. We matched all 35 selected questions to their source
UCLES TSA papers from 2019, 2020 and 2021, verified the answer options, and
transcribed the corresponding published keys into item_manifest.json.
That manifest records each source year, question number, choice, source URL,
workbook revision and question-text SHA-256 fingerprint.

The package downloads the source workbook to your computer during preparation.
It does not bundle copies of the complete exam papers or author workbook.
The downloadable workbook is about 46 KB; internet access is needed only for
that first preparation step. There is a manual-download fallback.

The default set is the study's 35 original questions, not its 35 newly authored
questions. It gives five items per criterion, keeps the CPU workload modest,
and uses independently published answer keys.

POINT DISTRIBUTION

Identifying the main conclusion                     5 questions x 1 point = 5
Drawing a conclusion                               5 questions x 1 point = 5
Identifying an assumption                          5 questions x 1 point = 5
Assessing the impact of additional evidence         5 questions x 1 point = 5
Detecting reasoning errors                         5 questions x 1 point = 5
Matching arguments                                 5 questions x 1 point = 5
Applying principles                                5 questions x 1 point = 5
TOTAL                                             35 questions; maximum 35

A correct selected option earns 1 point; an incorrect or unextractable option
earns 0 in the primary automatic score. There is no negative marking.
Total accuracy = correct answers / 35.
Criterion accuracy = correct answers for that criterion / 5.

The exact answer letters have unequal frequencies: A=10, B=10, C=4, D=3, E=8.
Uniform random guessing has expected accuracy 20%, while always choosing A or
B achieves 10/35 = 28.6%. The report includes both reference baselines.
Exceeding chance alone does not establish broad reasoning or critical thinking.
Do not invent a universal pass mark or apply Oxford admissions conversion tables
to this selected 35-item assessment.

The report includes a descriptive Wilson interval and a binomial upper-tail
calculation against independent uniform guesses. These have explicit assumptions;
they do not turn public, selected questions into a representative population
sample. Five questions per category are too few for precise category rankings.

HOW THE RUNNER WORKS

1. Reads only the prepared, fingerprint-checked question set.
2. Verifies your adapter_config.json and adapter weight files.
3. Loads the tokenizer/config from your existing Hugging Face cache.
4. Preflights every COMPLETE prompt before any model generation. Nothing is
   silently shortened to fit the input limit.
5. Loads your cached base weights and trained LoRA adapter directly on CPU.
6. Runs one fresh conversation per question, with greedy decoding.
7. Saves each raw response immediately; an interruption can be resumed.
8. Scores choices and creates a ZIP containing the evidence for review.

This is direct inference. You do not need web_app.py running. Stop the old web
server first with Ctrl+C in its terminal to avoid holding two model copies in RAM.
The base model supplies the underlying weights for the trained adapter; it is
not evaluated as a separate comparison condition.

The system prompt is "detailed thinking on" by default, matching your trained
project's control phrase. This phrase is a prompt instruction, not evidence of
an internal cognitive process. You may predeclare --mode off instead, but use
one chosen mode consistently and do not select a favorable mode after scoring.

The model is asked to put its answer first and then give a short explanation.
The answer key, source question number and criterion label are never sent to it.

The default budget is 2048 input tokens and 512 newly generated tokens, with
four CPU threads. The full input is checked against the limit, not truncated.
This input limit is independent of the small limits in your web_app.py server.
No longer-sequence retraining is required to run inference with this setting.
If all inputs exceed a limit, preparation stops before generating answers.

Only missing cached model/tokenizer files require --allow-model-download.
Normally do not use that option: your model is already downloaded.

WINDOWS INSTRUCTIONS

See COMMANDS_WINDOWS_CMD.txt. Run each stage individually:
  prepare_tsa.py       download and verify the questions
  evaluate_tsa.py run --smoke
                      one technical check, excluded from the final assessment
  evaluate_tsa.py run  all 35 final questions

Use your existing .venv; do not reinstall the model environment.
The instructions use D:\MSCS213-Simulation - Copy because it appeared in your
latest summary. Replace the root path if your actual project folder differs.

WHAT TO SEND BACK

Upload results\tsa_final\Compile_Answer.zip to this chat. It contains:
  run_config.json            identity, versions, source/code fingerprints,
                             settings and question order
  responses.jsonl            raw response, exact prompt/model input, token
                             counts, finish reason, time and parsed choice
  questions_and_keys.json    reference questions and verified choices
  summary.json               primary answer-key results, per criterion
  item_results.json          each question's automatic 0/1 point allocation
  RESULTS.txt                readable points and percentages
  review_packet.txt          questions, reference choices and exact responses
  review_template.json       structured, initially pending review fields
  REVIEW_GUIDE.txt            instructions for the supplemental reviewer
  item_manifest.json         source mappings and fingerprints

No model weights, training dataset, cache or adapter weights are included.
The script creates the ZIP locally. It does not automatically send anything.
A partial run also creates exports and reports its missing question IDs.
Final accuracy is null/unavailable until all planned items have responses.

SUPPLEMENTAL EXPLANATION REVIEW

ChatGPT can inspect whether the explanation actually supports the selected
answer, follows the passage, invents information, or contains an invalid step.
ChatGPT can also audit an answer stated clearly outside the recognized format.
Those are documented reviewer judgments, not an infallible source of truth.

Explanation categories are supported / flawed / insufficient / missing / unclear,
with pending for unreviewed entries. They are project audit labels, not an
official TSA explanation scale, and they add no points to the 35-point TSA score.
A correct letter with a flawed justification remains distinguishable from a
correct letter with a supported justification.

If answer extraction is corrected through review, the audited answer-key score
is reported separately; summary.json and its primary score remain intact.
Use exact response excerpts and written reasons for judgments. When feasible,
have a person independently review the ambiguous cases and a sample of the
remaining responses. Report who performed the review and how disagreements
were handled.

REPRODUCIBILITY AND RESUMING

Resume requires identical questions, adapter fingerprints, code, tokenizer/base
revision, mode, budget, thread count, seed and package versions.
Each question with a saved response is skipped, including incorrect or invalid
answers. Invalid answers are not regenerated until a favorable answer appears.
Failed generations without a saved answer may resume; the error log is retained.
If settings change, use a separately named output folder and disclose the change.
Do not edit or fine-tune on the assessment questions before collecting results.

DIFFERENCES FROM THE JOURNAL STUDY

This is a scoped adaptation, not an exact replication:
  - 35 original TSA items, not all 70 TSA items and not the essay test.
  - English; the journal study administered Chinese.
  - One completion per item, not 150 rounds at five temperatures.
  - Your small fine-tuned Qwen model rather than the models in the paper.
  - Your project's system control phrase, an answer marker and a brief reason.
  - Explicit CPU token budgets rather than the original human time conditions.
  - Additional explanation audit, reported separately from the TSA score.

The test measures selected argument-analysis behavior. It does not directly
measure every aspect of reasoning, human dispositions, autonomous decisions,
or the causal faithfulness of the model's internal process.
Public question exposure during pretraining cannot be ruled out. Do not label
the questions guaranteed unseen merely because you did not fine-tune on them.

INTERPRETATION FOR YOUR PRESENTATION

Use: "On a 35-item English TSA subset covering seven argument-analysis skills,
the model answered X/35 correctly under the recorded settings. The explanation
review found [supported/flawed/etc. counts], with [review method] and [limitations]."

If results are poor, say it did not demonstrate reliable performance on these
tasks under the tested conditions. Do not conclude that the model can never reason.
If results are strong, describe the demonstrated skills; do not claim that tags,
long responses or above-chance accuracy prove a faithful internal thought process.

VALIDATION OF THIS PACKAGE

Automated tests cover question import, answer extraction, constant-answer and
invalid-answer behavior, incomplete results, resume safeguards, failure logs,
review validation and an end-to-end mock run.
Mock outputs are software test fixtures, not results from your trained model.
The actual trained adapter is on your PC and must be evaluated there.

SOURCES

Jiang, L., Cui, Y., Liu, R., & Luo, F. (2025). Evaluating the critical thinking of
large language models: Insights and limitations. Journal of Pacific Rim Psychology,
19. https://doi.org/10.1177/18344909251406111
Authors' materials:
https://github.com/Jiang-Liming/Evaluating-Critical-Thinking-of-LLM

Cambridge Assessment Admissions Testing. (2021). Thinking Skills Assessment
question guide. UCLES. Primary document hosted at:
https://pmt.physicsandmathstutor.com/download/Admissions/TSA/TSA%20Question%20Guide.pdf

UCLES question papers and answer keys for 2019, 2020 and 2021:
Exact URLs and byte fingerprints are recorded in item_manifest.json.
The primary UCLES documents are available through public archive mirrors.
