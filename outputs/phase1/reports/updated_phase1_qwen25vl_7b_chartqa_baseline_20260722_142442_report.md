# Phase 1 - Qwen2.5-VL on ChartQA - A Generic VQA Baseline

## Experiment

- Phase: `phase1`
- Experiment name: `updated_phase1_qwen25vl_chartqa_baseline`
- Run ID: `20260718_224918` `20260722_142442` and `20260722_151850`
- Dataset: `HuggingFaceM4/ChartQA`
- Split: `val`
- Model: `Qwen/Qwen2.5-VL-3B-Instruct` and `Qwen/Qwen2.5-VL-7B-Instruct`
- Maximum samples: `200`
- Maximum new tokens: `64`
- Sampling enabled: `False`
- Number of beams: `1`
- Random seed: `42`
- Device map: `auto`
- Torch data type: `bfloat16`
- 4-bit quantization: `False`
- Minimum visual pixels: `200704`
- Maximum visual pixels: `1003520`

## Objective

This experiment establishes a generic visual question answering baseline using Qwen2.5-VL on ChartQA before transitioning to the medical multimodal visual question answering stage of the project.

Phase 1 is a technical and comparative baseline stage. Its purpose is to verify that the complete VQA pipeline works correctly on an established non-medical visual reasoning dataset before applying related evaluation principles to capsule endoscopy images.

Unlike chart-specialized models such as Pix2Struct, DePlot, and MatCha, Qwen2.5-VL is a general-purpose vision-language model. This experiment therefore evaluates whether a general multimodal model can interpret chart images, understand natural-language questions, perform visual and numerical reasoning, and directly generate final answers.

The initial run was performed using the smaller 3B-weights model. To enable a more meaningful comparison with similarly sized general-purpose models, such as InternVL2.5-8B, the following updates were introduced:

- replaced the initial Qwen2.5-VL-3B checkpoint with Qwen2.5-VL-7B-Instruct;
- executed the experiment on an NVIDIA A100 GPU in Google Colab;
- loaded the 7B model without 4-bit quantization, using BF16 precision;
- retained the same 200-example ChartQA subset, random seed, prompt, image-resolution limits, and generation settings for comparability;
- improved answer normalization while preserving decimal values and removing thousands separators;
- updated exact_match to recognize the gold answer when it appears within a longer generated response;
- added configurable relative and absolute numeric tolerances for approximate numerical answers;
- added equivalent handling of decimal and percentage representations, such as 0.71 and 71%;
- treated units such as million, billion, currencies, and percentage symbols as output formatting rather than as mismatches;
- implemented the evaluation logic through reusable, pure, and declarative helper functions;
- recorded execution metadata, inference time, model precision, quantization status, and inference errors.

## Scope

Qwen2.5-VL is evaluated as a generic vision-language baseline rather than as a model specialized specifically for charts or medical images.

Although Qwen2.5-VL can process a broad range of visual inputs, performance on ChartQA does not directly demonstrate clinical suitability. Chart images contain structured elements such as axes, legends, labels, bars, lines, and explicit numerical values, whereas capsule endoscopy images contain anatomical structures and visual clinical findings.

The principal outcomes of this Phase 1 experiment are:

- validating the dataset loading, inference, evaluation, and reporting pipeline;
- establishing a reproducible generic VQA baseline;
- evaluating visual, textual, structured, and numerical reasoning;
- comparing a general-purpose vision-language model with chart-specialized models;
- identifying common failure patterns in multimodal answer generation;
- preserving a consistent evaluation methodology across Phase 1 experiments;
- identifying design principles that may later support the medical MMVQA pipeline.

## Method

The experiment uses a declarative configuration object and modular helper functions to improve reproducibility and consistency across the Phase 1 baselines.

The configuration defines:

- the dataset and evaluation split;
- the model checkpoint;
- the selected sample count;
- the visual resolution limits;
- the numerical precision;
- the device placement strategy;
- the generation parameters;
- the random seed;
- the output directories.

Qwen2.5-VL is evaluated as a direct end-to-end visual question answering model.

For each ChartQA example, the model receives:

1. the chart image;
2. the associated natural-language question;
3. an instruction requesting only the final answer.

The image and question are represented as a multimodal conversation using the Qwen2.5-VL chat template. The processor converts the textual and visual inputs into model-ready tensors. The model then performs visual interpretation, multimodal reasoning, and answer generation within a single inference pipeline.

The model does not explicitly generate an intermediate table and does not use a separate external reasoning model.

The effective inference pipeline is:

`chart image + question + answer instruction -> Qwen2.5-VL -> final answer`

The remaining pipeline components perform:

- reproducible sample selection;
- RGB image conversion;
- multimodal prompt construction;
- visual input processing;
- deterministic answer generation;
- prompt-token removal;
- answer decoding;
- answer normalization;
- exact and numeric comparison;
- error collection;
- result serialization;
- aggregate metric computation.

## Generation Strategy

The model is evaluated using deterministic generation:

- `do_sample = False`
- `num_beams = 1`
- `max_new_tokens = 64`

The prompt instructs the model to answer using only information visible in the chart and to return only the final answer without an explanation.

This format reduces additional conversational text that could negatively affect Exact Match evaluation.

## Evaluation Metrics

Two primary evaluation metrics are used.

### 1. Exact Match Accuracy

The generated answer and the ground-truth answer are normalized and compared directly.

This metric is strict. Semantically equivalent answers may be counted as incorrect when their normalized textual forms remain different.

### 2. Numeric Match Accuracy

A numeric value is extracted from both the generated answer and the ground-truth answer and compared using the tolerance defined by the evaluation pipeline.

This metric provides additional information for questions where the model identifies the correct numerical value but formats the answer differently.

Inference errors are retained in the result set and counted as incorrect predictions. This ensures that the reported accuracy is calculated over the complete selected evaluation sample and remains comparable across models.

The Numeric Match score provides a complementary view of performance on answers containing numerical information. Its lower value should not be interpreted as contradicting the Exact Match result, because the two metrics evaluate different answer properties. Exact Match also evaluates categorical, textual, colour-based, and yes-or-no answers, whereas Numeric Match is relevant primarily when comparable numerical values can be extracted from both the prediction and the reference answer. Consequently, Numeric Match should be treated as a diagnostic metric rather than as a direct replacement for overall accuracy.


## Results

Qwen/Qwen2.5-VL-3B-Instruct #1:

- Number of evaluated examples: `200`
- Successful predictions: `200`
- Errors during inference: `0`
- Error rate: `0.0000`
- Exact Match Accuracy: `0.71`
- Numeric Match Accuracy: `0.545`
- Mean inference time: `3.76 seconds`
- Timestamp: 2026-07-18T22:45:21.227759-04:00

Qwen/Qwen2.5-VL-3B-Instruct #2:

- Number of evaluated examples: `200`
- Successful predictions: `200`
- Errors during inference: `0`
- Error rate: `0.0000`
- Exact Match Accuracy: `0.845`
- Numeric Match Accuracy: `0.67`
- Mean inference time: `4.18 seconds`
- Timestamp: 2026-07-22T15:40:28.733309-04:00


Qwen/Qwen2.5-VL-7B-Instruct:

- Number of evaluated examples: `200`
- Successful predictions: `200`
- Errors during inference: `0`
- Error rate: `0.0000`
- Exact Match Accuracy: `0.8850`
- Numeric Match Accuracy: `0.7050`
- Mean inference time: `0.3252 seconds`
- Timestamp: 2026-07-22T14:27:26.520877-04:00

## Output Files

Qwen/Qwen2.5-VL-3B-Instruct #1:

- Configuration JSON: `outputs/phase1/config/phase1_qwen25vl_chartqa_baseline_20260718_224918_config.json`
- Results CSV: `outputs/phase1/results/phase1_qwen25vl_chartqa_baseline_20260718_224918_results.csv`
- Metrics JSON: `outputs/phase1/results/phase1_qwen25vl_chartqa_baseline_20260718_224918_metrics.json`

Qwen/Qwen2.5-VL-3B-Instruct #2:

- Configuration JSON: `outputs/phase1/config/phase1_qwen25vl_3b_chartqa_baseline_20260722_151850_config.json`
- Results CSV: `outputs/phase1/results/phase1_qwen25vl_3b_chartqa_baseline_20260722_151850_results.csv`
- Metrics JSON: `outputs/phase1/results/phase1_qwen25vl_3b_chartqa_baseline_20260722_151850_metrics.json`

Qwen/Qwen2.5-VL-7B-Instruct:

- Configuration JSON: `outputs/phase1/config/updated_phase1_qwen25vl_7b_chartqa_baseline_20260722_142442_config.json`
- Results CSV: `outputs/phase1/results/updated_phase1_qwen25vl_7b_chartqa_baseline_20260722_142442_results.csv`
- Metrics JSON: `outputs/phase1/results/updated_phase1_qwen25vl_7b_chartqa_baseline_20260722_142442_metrics.json`

## Interpretation

The Qwen2.5-VL-7B experiment was completed on the same selected 200-example ChartQA validation subset using the non-quantized checkpoint on an A100 GPU with BF16 precision. The experiment used deterministic generation, the same visual-resolution limits, and the same multimodal prompt structure across all examples. All 200 samples were processed successfully, with no inference errors.
Following improvements to the Exact Match and Numeric Match functions, Qwen2.5-VL-7B achieved an `Exact Match Accuracy of 88.5%` and a `Numeric Match Accuracy of 70.5%`. This corresponds to 177 exact matches out of 200 examples. During the qualitative error analysis, at least one incorrectly annotated ground-truth label was also identified. In that example, the apparent model error was caused by the dataset annotation rather than by the prediction. This means that the reported 88.5% score is slightly conservative; correcting one confirmed mislabeled example would raise the effective Exact Match result from 177/200 to 178/200, or 89.0%. Additional manual inspection may reveal other annotation issues or semantically valid predictions that remain difficult to capture through automatic matching. The revised evaluation logic better recognizes answers that are semantically or numerically correct but expressed in a different surface form, such as responses containing additional wording or equivalent representations such as 0.71 and 71%.

The importance of the evaluation changes is demonstrated by the two Qwen2.5-VL-3B runs. The first run, evaluated with the original metric functions, `obtained 71.0% Exact Match Accuracy` and `54.5% Numeric Match Accuracy`. After the metric functions were updated, the second 3B run `obtained 84.5% Exact Match Accuracy` and `67.0% Numeric Match Accuracy` under the same general experimental configuration.

| Model and evaluation version     | Exact Match | Numeric Match | Exact matches |
| -------------------------------- | ----------: | ------------: | ------------: |
| Qwen2.5-VL-3B, original metrics  |       71.0% |         54.5% |       142/200 |
| Qwen2.5-VL-3B, corrected metrics |       84.5% |         67.0% |       169/200 |
| Qwen2.5-VL-7B, corrected metrics |       88.5% |         70.5% |       177/200 |

For the 3B model, the corrected evaluation increased Exact Match Accuracy by 13.5 percentage points, equivalent to 27 additional answers being recognized as correct. Numeric Match Accuracy increased by 12.5 percentage points, corresponding to 25 additional numeric matches. Since the model checkpoint, dataset subset, prompt structure, and deterministic generation settings remained unchanged, this improvement should be interpreted primarily as the removal of false negatives introduced by the earlier evaluation logic, rather than as an improvement in the model itself.

The original Exact Match function was overly sensitive to response formatting. A prediction could be conceptually correct but still be marked as incorrect when the model included short explanatory wording, used an alternative capitalization or formatting style, or returned the correct answer as part of a longer phrase. Similarly, the original numeric comparison did not consistently recognize equivalent representations, particularly percentages and decimal values. The corrected functions therefore provide a more faithful estimate of answer correctness while preserving a consistent automated evaluation process.

Using the corrected metrics, the comparison between the two Qwen checkpoints becomes clearer. `Qwen2.5-VL-7B outperformed Qwen2.5-VL-3B by 4.0 percentage points in Exact Match Accuracy` and 3.5 percentage points in Numeric Match Accuracy. The 7B model produced 177 exact matches, compared with 169 for the 3B model, resulting in eight additional correct answers and eight fewer Exact Match failures. It also produced seven additional numeric matches. These results indicate that increasing the model size from 3B to 7B provides a measurable, although moderate, improvement on the selected ChartQA subset.

The 3B checkpoint nevertheless remains highly competitive. An Exact Match Accuracy of 84.5% shows that the smaller model already performs strongly on chart perception, question understanding, numerical reasoning, and concise answer generation. The 7B checkpoint provides the stronger overall result, while the 3B checkpoint may remain attractive when memory requirements, computational cost, or deployment constraints are important.

The inference-time results should be interpreted separately from model accuracy. The two 3B runs recorded mean inference times of 3.76 seconds and 4.18 seconds per example. This difference is more likely to reflect normal runtime variability, GPU state, caching, warm-up effects, or surrounding Colab conditions than a consequence of the revised evaluation functions. The metric changes affect scoring after prediction and should not materially change the model’s actual generation speed. Similarly, the substantially lower mean inference time measured for the 7B run should not be treated as evidence that the larger model is inherently faster unless all models are benchmarked under a controlled and identical timing procedure.

Overall, Qwen2.5-VL-7B establishes a strong general-purpose multimodal baseline for Phase 1. It demonstrates effective chart perception, textual comprehension, structured visual reasoning, and answer generation within a single end-to-end model. The corrected comparison also confirms that the 7B checkpoint performs better than the 3B checkpoint, but that the performance gain is moderate rather than proportional to the increase in model size.

The 7B result provides an appropriate reference point for the upcoming InternVL2.5-8B experiment. Since InternVL has a broadly similar parameter scale but belongs to a different vision-language model family, the comparison can help determine whether differences arise from model size, architecture, visual encoding, multimodal training, or instruction-following behaviour. However, the strong ChartQA result should not be interpreted as direct evidence of competence on capsule endoscopy images or clinical reasoning tasks. Medical evaluation, grounding, calibration, and domain-specific adaptation must still be assessed independently.
