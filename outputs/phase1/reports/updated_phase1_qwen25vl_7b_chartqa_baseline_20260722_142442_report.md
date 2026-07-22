# Phase 1 - Qwen2.5-VL-7B on ChartQA - A Generic VQA Baseline

## Experiment

- Phase: `phase1`
- Experiment name: `updated_phase1_qwen25vl_7b_chartqa_baseline`
- Run ID: `20260722_142442`
- Dataset: `HuggingFaceM4/ChartQA`
- Split: `val`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
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

- Number of evaluated examples: `200`
- Successful predictions: `200`
- Errors during inference: `0`
- Error rate: `0.0000`
- Exact Match Accuracy: `0.8850`
- Numeric Match Accuracy: `0.7050`
- Mean inference time: `0.3252 seconds`

## Output Files

- Configuration JSON: `outputs/phase1/config/updated_phase1_qwen25vl_7b_chartqa_baseline_20260722_142442_config.json`
- Results CSV: `outputs/phase1/results/updated_phase1_qwen25vl_7b_chartqa_baseline_20260722_142442_results.csv`
- Metrics JSON: `outputs/phase1/results/updated_phase1_qwen25vl_7b_chartqa_baseline_20260722_142442_metrics.json`

## Interpretation

The Qwen2.5-VL-7B experiment was completed on the same selected 200-example ChartQA validation subset using the non-quantized model on an A100 GPU with BF16 precision as explained above. The evaluation used deterministic generation, identical visual-resolution limits, and the same multimodal prompt structure across all examples. All 200 samples were processed successfully, with no inference errors.

After improving the Exact Match and Numeric Match evaluation functions, which had failed to recognize some correct answers because of longer response phrasing or equivalent numerical formats—for example, the model achieved an Exact Match Accuracy of 88.5% and a Numeric Match Accuracy of 70.5%. This corresponds to 177 exact matches out of 200 evaluated examples. During the error analysis, I also identified at least one incorrectly annotated ground-truth label in the dataset, meaning that one apparent model error was actually caused by a dataset-labeling issue. Overall, the high Exact Match score indicates that the model generally interpreted the charts correctly and produced answers consistent with the expected short-answer format.

The absence of inference errors confirms that the non-quantized 7B checkpoint can be executed reliably in the selected A100 environment. The mean inference time of 0.3252 seconds per example also demonstrates that the larger model can provide efficient inference when sufficient GPU resources are available.

These updated results substantially change the interpretation of the model. Qwen2.5-VL-7B demonstrates strong performance on the selected ChartQA subset and appears capable of combining chart perception, textual understanding, structured visual reasoning, and short-answer generation within a single end-to-end model. 
However, the result remains specific to the selected 200-example validation subset and the current evaluation protocol. Additional analysis should examine the remaining incorrect answers, including semantically equivalent responses that may still fail strict Exact Match, such as alternative colour descriptions, formatting differences, percentage representations, abbreviations, or valid synonymous expressions.

Overall, Qwen2.5-VL-7B establishes a strong general-purpose multimodal baseline for Phase 1. It is also an appropriate reference point for the upcoming InternVL2.5-8B experiment, which will determine whether a similarly sized model from a different vision-language family provides complementary strengths or different failure patterns. The strong ChartQA result supports retaining Qwen2.5-VL-7B as a candidate for subsequent medical evaluation, while recognizing that performance on structured charts does not directly establish competence on capsule endoscopy images or clinical reasoning tasks.
