# Phase 1 - InternVL2.5-8B on ChartQA - A Generic VQA Baseline

## Experiment

- Phase: `phase1`
- Experiment name: `phase1_internvl25_8b_chartqa_baseline`
- Run ID: `20260724_202448`
- Dataset: `HuggingFaceM4/ChartQA`
- Split: `val`
- Model: `OpenGVLab/InternVL2_5-8B`
- Maximum samples: `200`
- Maximum new tokens: `64`
- Sampling enabled: `False`
- Number of beams: `1`
- Random seed: `42`
- Device placement: `cuda`
- Torch data type: `bfloat16`
- Weight quantization enabled: `False`
- InternVL image tile size: `448 x 448`
- Maximum number of image tiles: `12`
- Global thumbnail enabled: `True`
- Flash Attention enabled: `False`
- Trust remote model code: `True`

## Objective

This experiment establishes a generic visual question answering baseline using InternVL2.5-8B on ChartQA before transitioning to the medical multimodal visual question answering stage of the project.

Phase 1 is a technical and comparative baseline stage. Its purpose is to verify that the complete VQA pipeline operates correctly on an established non-medical visual reasoning dataset before applying related evaluation principles to capsule endoscopy images.

Unlike chart-specialized architectures such as Pix2Struct, DePlot, and MatCha, InternVL2.5-8B is a general-purpose vision-language model. This experiment evaluates whether a general multimodal model can interpret chart images, understand natural-language questions, perform visual and numerical reasoning, and directly generate concise final answers.

The experiment also provides a general-purpose multimodal baseline that can be compared with the previously evaluated Qwen2.5-VL models under the same ChartQA sample selection and evaluation methodology.

## Scope

InternVL2.5-8B is evaluated as a generic vision-language baseline rather than as a model specifically specialized for chart understanding or medical images.

Although InternVL2.5-8B can process a broad range of visual inputs, strong performance on ChartQA does not directly demonstrate clinical suitability. Chart images contain structured visual elements such as axes, legends, labels, bars, lines, and explicit numerical values. Capsule endoscopy images instead contain anatomical structures, tissue characteristics, and visual clinical findings that require different forms of representation and interpretation.

The principal outcomes of this Phase 1 experiment are:

- validating the dataset loading, image preprocessing, inference, evaluation, and reporting pipeline;
- establishing a reproducible general-purpose VQA baseline;
- evaluating visual, textual, structured, and numerical reasoning;
- comparing InternVL2.5-8B with Qwen2.5-VL and chart-specialized models;
- identifying common failure patterns in multimodal answer generation;
- preserving a consistent evaluation methodology across Phase 1 experiments;
- evaluating the reliability and execution speed of a non-quantized 8B model;
- identifying design principles that may later support the medical MMVQA pipeline.

## Method

The experiment uses a declarative configuration object and modular helper functions to improve reproducibility and maintain consistency across the Phase 1 baselines.

The configuration defines:

- the dataset and evaluation split;
- the model checkpoint;
- the selected sample count;
- the image tile size;
- the maximum number of visual tiles;
- whether a global image thumbnail is included;
- the numerical precision;
- the device placement;
- the generation parameters;
- the random seed;
- the output directories.

InternVL2.5-8B is evaluated as a direct end-to-end visual question answering model.

For every ChartQA example, the model receives:

1. the chart image;
2. the associated natural-language question;
3. an instruction requesting only the final answer.

The image is first converted to RGB format. InternVL's dynamic visual preprocessing procedure then selects a tile grid based on the aspect ratio of the original chart. Each tile is resized to `448 x 448` pixels and normalized using ImageNet statistics.

When an image is divided into multiple tiles and global thumbnail processing is enabled, an additional square thumbnail representing the complete image is appended to the visual input. This allows the model to receive both local high-resolution chart regions and global chart context.

The prompt includes InternVL's required `<image>` token and is passed to the model together with the generated visual tensors. The tokenizer processes the textual prompt, while the model's `chat()` method performs multimodal interpretation and answer generation.

The model does not explicitly generate an intermediate table and does not use a separate external reasoning model.

The effective inference pipeline is:

`chart image -> dynamic visual tiling + global thumbnail -> InternVL2.5-8B + question -> final answer`

The remaining pipeline components perform:

- reproducible sample selection;
- RGB image conversion;
- dynamic image tiling;
- image tensor normalization;
- multimodal prompt formatting;
- deterministic answer generation;
- answer normalization;
- exact and numeric comparison;
- per-example inference-time measurement;
- error collection;
- result serialization;
- aggregate metric computation.

## Generation Strategy

The model is evaluated using deterministic generation:

- `do_sample = False`
- `num_beams = 1`
- `max_new_tokens = 64`

The prompt instructs the model to answer using only the information visible in the chart and to return only the final answer without an explanation.

This format reduces additional conversational text and makes the model output more compatible with the short-answer structure expected by ChartQA.

## Evaluation Metrics

Two primary evaluation metrics are used.

### 1. Exact Match Accuracy

The `exact_match` metric applies different comparison rules depending on the type of ground-truth answer.

For numeric ground-truth answers, the prediction is considered correct when at least one extracted numerical representation matches the ground-truth value within the configured relative or absolute tolerance.

Equivalent percentage representations are also considered. For example, `71%` and `0.71` can be recognized as equivalent numerical answers.

For non-numeric answers, the complete normalized ground-truth answer must occur within the normalized model prediction. Word-boundary padding prevents partial matches such as matching `yes` inside `yesterday`.

Therefore, the metric retains the project-wide name `exact_match`, but it is more flexible than a traditional strict string-equality metric. It recognizes correct answers embedded in longer generated phrases and equivalent numerical formats.

### 2. Numeric Match Accuracy

The Numeric Match metric extracts all numerical representations from the generated answer and the ground-truth answer.

A prediction is considered numerically correct when at least one predicted value matches at least one ground-truth value under the configured numerical policy:

- Relative tolerance: `0.05`
- Absolute tolerance: `0.05`

Percentage and decimal representations are treated as potentially equivalent.

Numeric Match is reported separately because it applies only to numerical content and provides additional insight into the model's chart-value extraction and quantitative reasoning performance.

### Error Treatment

Inference errors are retained in the result set and counted as incorrect predictions. This ensures that accuracy is calculated over the complete selected evaluation sample and remains comparable across all Phase 1 experiments.

In this execution, all `200` selected examples were processed successfully, with no inference errors.

## Results

- Number of evaluated examples: `200`
- Successful predictions: `200`
- Errors during inference: `0`
- Error rate: `0.0000`
- Exact Match correct predictions: `179 / 200`
- Predictions not satisfying Exact Match: `21 / 200`
- Exact Match Accuracy: `0.8950` (89.5%)
- Numeric Match correct predictions: `141 / 200`
- Predictions not satisfying Numeric Match: `59 / 200`
- Numeric Match Accuracy: `0.7050` (70.5%)
- Mean inference time per example: `0.5204 seconds`
- Approximate total inference time: `104.09 seconds`

## Result Summary

InternVL2.5-8B achieved an Exact Match Accuracy of `89.5%`, corresponding to `179` correct predictions from `200` evaluated ChartQA examples.

A total of `21` predictions did not satisfy the Exact Match evaluation policy.

The model achieved a Numeric Match Accuracy of `70.5%`, corresponding to `141` numerically matched examples. Numeric Match is lower than the overall Exact Match score because the complete evaluation set also contains textual and categorical answers, and because some answers can satisfy the normalized text-containment rule without producing a numerical match.

All `200` examples were processed successfully. The absence of inference errors confirms that the non-quantized InternVL2.5-8B checkpoint can be executed reliably in the selected A100 environment using `bfloat16` precision.

The mean inference time was `0.5204 seconds` per example, producing an approximate total inference time of `104.09 seconds` for the selected evaluation subset.

## Output Files

- Configuration JSON: `outputs/phase1/configs/phase1_internvl25_8b_chartqa_baseline_20260724_202448_config.json`
- Results CSV: `outputs/phase1/results/phase1_internvl25_8b_chartqa_baseline_20260724_202448_results.csv`
- Metrics JSON: `outputs/phase1/results/phase1_internvl25_8b_chartqa_baseline_20260724_202448_metrics.json`

## Interpretation

The model `OpenGVLab/InternVL2_5-8B` is a general-purpose vision-language model evaluated as a direct ChartQA answer-generation baseline.

It receives dynamically processed chart-image tiles and an associated natural-language question and generates the final textual answer without producing an explicit intermediate table or using a separate text-based reasoning model.

The Exact Match Accuracy of `89.5%` indicates that the model interpreted the large majority of the selected charts correctly and produced answers compatible with the expected short-answer format. Only `21` out of `200` predictions failed the configured Exact Match policy.

The Numeric Match Accuracy of `70.5%` indicates strong, although lower, performance on answers containing numerical information. Detailed error analysis is still necessary to distinguish between incorrect chart interpretation, arithmetic errors, visual-value extraction failures, formatting differences not handled by the evaluation policy, ambiguous questions, and potential dataset-label issues.

This experiment preserves the same ChartQA sample selection, answer-normalization rules, Exact Match policy, Numeric Match policy, error treatment, and reporting structure used for the other Phase 1 baselines. Only the model-loading, image-processing, prompt-construction, and inference components are adapted specifically for InternVL2.5-8B.

The result provides evidence that InternVL2.5-8B is a strong general-purpose multimodal baseline for structured visual reasoning. However, this result should not be interpreted as evidence of performance on capsule endoscopy images or as evidence of clinical reliability.

Later medical evaluation stages must independently assess the model's ability to recognize clinical findings, provide visually grounded answers, avoid unsupported conclusions, communicate uncertainty, and operate safely within the intended medical decision-support scope.

The experiment validates the complete InternVL inference and evaluation pipeline and provides a directly comparable result for the other generic and chart-specialized models evaluated during Phase 1.
