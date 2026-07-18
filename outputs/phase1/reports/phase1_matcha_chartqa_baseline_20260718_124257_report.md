# Phase 1 - DePlot on ChartQA - A Generic VQA Baseline

## Experiment

- Phase: `phase1`
- Experiment name: `phase1_matcha_chartqa_baseline`
- Run ID: `20260718_124257`
- Dataset: `HuggingFaceM4/ChartQA`
- Split: `val`
- Model: `google/matcha-chartqa`
- Max samples: `200`
- Max new tokens: `64`
- Seed: `42`
- Device: `cuda`

## Objective

This experiment establishes an initial generic visual question answering baseline using MatCha pre-trained model tested on a chart-based VQA dataset before moving to medical capsule endoscopy data. Phase 1 is a technical baseline stage. Its purpose is to test whether the VQA pipeline works correctly on established non-medical datasets before moving to medical data.

## Scope

MatCha is not expected to become a candidate for the medical VQA stage. Capsule endoscopy images do not contain charts, tables, axes, or explicit numerical structures. Therefore, strong performance on ChartQA would not demonstrate suitability for medical images, while weak chart performance would make transfer to the medical domain even less promising.
The main outcomes of this Phase 1 experiment are:

- validating the dataset, inference, evaluation, and reporting pipeline;
- establishing reproducible generic VQA baselines;
- comparing direct and modular reasoning approaches;
- showing that model specialization must match the target image domain;
- evaluate numerical and structured visual reasoning;
- identify design principles that might later be reused in the medical system.


## Method

The experiment uses a declarative configuration object and modular helper functions to ensure reproducibility and consistency with the other Phase 1 baselines. The configuration defines the dataset, model checkpoint, evaluation split, sample limit, generation parameters, random seed, execution device, and output paths.

MatCha is evaluated as a direct visual question answering model. For each ChartQA example, the model receives the chart image together with the associated natural-language question. The processor converts both inputs into the representation required by the Pix2Struct-based architecture, and MatCha directly generates the final answer.

Unlike the DePlot pipeline, MatCha does not first convert the chart into a linearized table and does not use a separate text-based reasoner. Visual understanding, structured reasoning, and answer generation are performed within the same end-to-end model.

The remaining pipeline functions handle reproducible sample selection, image preprocessing, inference, answer normalization, metric computation, error collection, and output serialization.

The effective inference pipeline is:

`chart image + question -> MatCha -> final answer`

## Evaluation Metrics

Two simple metrics were used:

1. **Exact Match Accuracy**
   The generated answer and ground-truth answer are normalized and compared directly.

2. **Numeric Match Accuracy**
   The first numeric value is extracted from both the generated answer and the ground-truth answer and compared using a small tolerance.

## Results

- Number of evaluated examples: `200`
- Errors during inference: `0`
- Exact Match Accuracy: `0.5050`
- Numeric Match Accuracy: `0.3750`

## Output Files

- Results CSV: `outputs/phase1/results/phase1_matcha_chartqa_baseline_20260718_124257_results.csv`
- Metrics JSON: `outputs/phase1/results/phase1_matcha_chartqa_baseline_20260718_124257_metrics.json`

The model google/matcha-chartqa is a direct ChartQA answer-generation model based on the Pix2Struct architecture. It receives the chart image and the associated question as inputs and generates the final answer without producing an intermediate table or using an external text-based reasoner.

This experiment preserves the same dataset selection, evaluation functions, normalization rules, metrics, and reporting format used for the Pix2Struct and DePlot baselines. Only the model-loading and prediction components are adapted to MatCha's direct end-to-end inference approach.

The results should be interpreted as a direct visual question answering baseline for structured chart reasoning rather than as a modular chart-to-table reasoning pipeline.

