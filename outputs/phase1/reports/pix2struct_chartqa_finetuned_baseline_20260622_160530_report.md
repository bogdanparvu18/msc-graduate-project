# Phase 1 - Pix2struct on ChartQA - Generic VQA Baseline

## Experiment

- Phase: `phase1`
- Experiment name: `pix2struct_chartqa_finetuned_baseline`
- Run ID: `20260622_160530` and `20260624_160513`
- Dataset: `HuggingFaceM4/ChartQA`
- Split: `val`
- Model: `google/pix2struct-chartqa-base`
- Max samples: `50`
- Max new tokens: `64`
- Seed: `42`
- Device: `cuda`

## Objective

This experiment establishes an initial generic visual question answering baseline using a chart-based VQA dataset before moving to medical capsule endoscopy data. Phase 1 is a technical baseline stage. Its purpose is to test whether the VQA pipeline works correctly on established non-medical datasets before moving to medical data.

## Method

The experiment uses a declarative configuration object and modular functional-style helper functions. Pix2Struct answers questions directly from charts. The configuration defines the dataset, model, split, generation parameters, random seed, device, and output paths. The pipeline functions handle data loading, sample selection, inference, answer normalization, metric computation, and output serialization.

## Scope

This model is not expected to become a candidate to medical VQA models. Capsule-endoscopy images do not contain charts, tables, axes, or numerical structures. Therefore, strong chart performance would not prove medical suitability, and weak chart performance makes transfer even less promising.

The main outcomes of this Phase 1 experiment are:

- validating the dataset, inference, evaluation, and reporting pipeline;
- establishing reproducible generic VQA baselines;
- comparing direct and modular reasoning approaches;
- showing that model specialization must match the target image domain;
- evaluate numerical and structured visual reasoning;
- identify design principles that might later be reused in the medical system.

## Evaluation Metrics

Two simple metrics were used:

1. **Exact Match Accuracy**  
   The generated answer and ground-truth answer are normalized and compared directly.

2. **Numeric Match Accuracy**  
   The first numeric value is extracted from both the generated answer and the ground-truth answer and compared using a small tolerance.

## Results

### First run had this [config file](/outputs/phase1/configs/pix2struct_chartqa_finetuned_baseline_20260622_160530_config.json)
- Number of evaluated examples: `50`
- Errors during inference: `0`
- Exact Match Accuracy: `0.1400`
- Numeric Match Accuracy: `0.0800`

### Second run had this [config file](/outputs/phase1/configs/updated_pix2struct_chartqa_finetuned_baseline_20260624_160513_config.json)
- Number of evaluated examples: `200`
- Errors during inference: `0`
- Exact Match Accuracy: `0.455`
- Numeric Match Accuracy: `0.355`

## Output Files

- Results CSV: `outputs/phase1/results/pix2struct_chartqa_finetuned_baseline_20260622_160530_results.csv`
- Metrics JSON: `outputs/phase1/results/pix2struct_chartqa_finetuned_baseline_20260622_160530_metrics.json`
- Results #2 CSV: `outputs/phase1/results/updated_pix2struct_chartqa_finetuned_baseline_20260624_160513_results.csv`
- Metrics #2 JSON: `outputs/phase1/results/updated_pix2struct_chartqa_finetuned_baseline_20260624_160513_metrics.json`

## Pix2Struct conclusion

The initial Pix2Struct evaluation on the first 50 ChartQA validation examples produced no runtime errors, confirming that the inference pipeline was stable. However, the results were very low, with 14% exact match and 8% numeric match. Since the processor was verified to run in VQA mode, manual inspection suggested that many errors were substantive prediction failures rather than formatting or preprocessing issues.

After switching to a randomized sample of 200 validation examples, performance improved to 45.5% exact match and 35.5% numeric match. This indicates that the first 50 examples were likely not representative of the validation distribution. Overall, Pix2Struct provides a usable but limited Phase 1 baseline, especially for numeric chart reasoning, and should be compared against stronger chart-oriented and general vision-language models.


## Notes

The model `google/pix2struct-chartqa-base` is already fine-tuned on ChartQA. Therefore, this experiment should be treated as a fine-tuned baseline rather than a strict zero-shot experiment.

The next step is to repeat the same pipeline with additional datasets and models, such as DVQA, DashboardQA, DePlot, Matcha, Qwen2.5-VL, InternVL, and LLaVA-style models.
