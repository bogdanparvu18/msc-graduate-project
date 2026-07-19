# Phase 1 - Qwen2.5-VL on ChartQA - A Generic VQA Baseline

## Experiment

- Phase: `phase1`
- Experiment name: `phase1_qwen25vl_chartqa_baseline`
- Run ID: `20260718_224918`
- Dataset: `HuggingFaceM4/ChartQA`
- Split: `val`
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
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

## Results

- Number of evaluated examples: `200`
- Successful predictions: `200`
- Errors during inference: `0`
- Error rate: `0.0000`
- Exact Match Accuracy: `0.7100`
- Numeric Match Accuracy: `0.5450`
- Mean inference time: `3.7666 seconds`

## Output Files

- Configuration JSON: `Not available`
- Results CSV: `outputs/phase1/results/phase1_qwen25vl_chartqa_baseline_20260718_224918_results.csv`
- Metrics JSON: `outputs/phase1/results/phase1_qwen25vl_chartqa_baseline_20260718_224918_metrics.json`

## Interpretation

The model `Qwen/Qwen2.5-VL-3B-Instruct` is a general-purpose vision-language model evaluated as a direct ChartQA answer-generation baseline.

It receives the chart image and associated question as multimodal inputs and generates the final textual answer without producing an explicit intermediate table or using a separate text-based reasoning model.

This experiment preserves the same dataset selection strategy, answer normalization rules, Exact Match metric, Numeric Match metric, error treatment, and reporting structure used for the other Phase 1 baselines. The model-loading, multimodal input preparation, prompt construction, and answer-decoding components are adapted specifically to Qwen2.5-VL.

The results should be interpreted as a generic multimodal visual reasoning baseline on structured chart images. They should not be interpreted as evidence of performance on capsule endoscopy images or as evidence of clinical reliability.

The later medical evaluation stages must independently assess the model's ability to recognize clinical findings, provide grounded answers, avoid unsupported conclusions, and operate safely within the intended medical decision-support scope.
