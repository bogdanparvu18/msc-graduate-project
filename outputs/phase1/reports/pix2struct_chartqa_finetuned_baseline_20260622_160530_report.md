# Phase 1 - Generic VQA Baseline

## Experiment

- Phase: `phase1`
- Experiment name: `pix2struct_chartqa_finetuned_baseline`
- Run ID: `20260622_160530`
- Dataset: `HuggingFaceM4/ChartQA`
- Split: `val`
- Model: `google/pix2struct-chartqa-base`
- Max samples: `50`
- Max new tokens: `64`
- Seed: `42`
- Device: `cuda`

## Objective

This experiment establishes an initial generic visual question answering baseline using a chart-based VQA dataset before moving to medical capsule endoscopy data.

## Method

The experiment uses a declarative configuration object and modular functional-style helper functions. The configuration defines the dataset, model, split, generation parameters, random seed, device, and output paths. The pipeline functions handle data loading, sample selection, inference, answer normalization, metric computation, and output serialization.

## Evaluation Metrics

Two simple metrics were used:

1. **Exact Match Accuracy**  
   The generated answer and ground-truth answer are normalized and compared directly.

2. **Numeric Match Accuracy**  
   The first numeric value is extracted from both the generated answer and the ground-truth answer and compared using a small tolerance.

## Results

- Number of evaluated examples: `50`
- Errors during inference: `0`
- Exact Match Accuracy: `0.1400`
- Numeric Match Accuracy: `0.0800`

## Output Files

- Results CSV: `outputs/phase1/results/pix2struct_chartqa_finetuned_baseline_20260622_160530_results.csv`
- Metrics JSON: `outputs/phase1/results/pix2struct_chartqa_finetuned_baseline_20260622_160530_metrics.json`

## First run conclusion

The initial Pix2Struct baseline on the ChartQA validation split with first 50 samples produced no runtime errors, which confirms that the inference pipeline executed successfully. However, the prediction quality was poor: exact-match accuracy was 0.14 and numeric-match accuracy was 0.08 on the first 50 validation examples. Manual inspection shows that many failures are not minor formatting or exact-match issues, but substantive prediction errors, including incorrect numerical values, unrelated dates, and semantically unrelated answers. This suggests that the main limitation is not code execution, but either weak model generalization on the selected validation examples or a possible Pix2Struct input/preprocessing mismatch. The result should therefore be treated as a weak initial baseline that requires verification through preprocessing checks, larger randomized evaluation, and comparison against stronger VLMs.

## Notes

The model `google/pix2struct-chartqa-base` is already fine-tuned on ChartQA. Therefore, this experiment should be treated as a fine-tuned baseline rather than a strict zero-shot experiment.

The next step is to repeat the same pipeline with additional datasets and models, such as DVQA, DashboardQA, DePlot, Qwen2.5-VL, InternVL, and LLaVA-style models.
