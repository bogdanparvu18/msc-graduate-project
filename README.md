# Multimodal Medical Question Answering over Capsule Endoscopy with Grounded Clinical Finding Interpretation and Guarded Decision Support

## Executive workflow summary 

A clinician or medical researcher asks a natural-language question about capsule-endoscopy evidence, and the system retrieves and reasons over capsule-endoscopy frames, short frame sequences, visual finding annotations, temporal context, metadata, and structured labels to produce an evidence-grounded answer, a clinical finding interpretation, and a guarded decision-support recommendation.

The system will perform the following: 

1. Understand the user’s natural-language medical question.
2. Retrieve relevant capsule-endoscopy evidence.
3. Fuse visual and structured medical evidence.
4. Reason over the evidence.
5. Return a grounded answer with confidence and evidence references.
6. Convert the answer into a clinical finding interpretation.
7. Map the finding interpretation to guarded decision-support outputs.
8. Apply medical safety checks before any recommendation:
   - confidence threshold
   - evidence sufficiency
   - clinical guardrails
   - human-clinician review
No autonomous diagnosis or treatment decision


## Problem statement
Capsule endoscopy produces large volumes of visual data that are difficult and time-consuming to review manually. A single examination may contain thousands of frames, while clinically important findings may appear only briefly, across a small number of frames, or in visually subtle forms. This creates a challenging workflow for clinicians, who must identify relevant frames, interpret possible abnormalities, compare findings across temporal context, and decide whether further review or action is needed.

Existing computer-aided systems can support classification or detection, but they often do not provide natural-language, evidence-grounded explanations. Generic multimodal models may describe visual content, but they are not automatically reliable for medical interpretation unless their answers are grounded in verified evidence, constrained by medical guardrails, and evaluated against expert-labeled data.

This project addresses that gap by proposing a medical multimodal question-answering system for capsule endoscopy. The system accepts natural-language medical questions, retrieves relevant capsule-endoscopy visual artifacts and structured evidence, produces grounded answers, identifies candidate clinical findings, and returns guarded decision-support recommendations. The goal is not to replace clinicians, but to support faster, more consistent, and more explainable interpretation of capsule-endoscopy evidence.

### Example questions:

 - Is there evidence of bleeding in this frame sequence?
 - Which frames support the finding?
 - Is the abnormality localized or persistent across time?
 - What visual evidence supports this interpretation?
 - Should this case be flagged for clinician review?
 - Is the evidence sufficient, or should the model abstain?

## Project workflow



####  Phase 1 — Establish the generic VQA baseline on DVQA & ChartQA
  This phase establishes the project’s initial multimodal reasoning baseline before moving into capsule-endoscopy data. The objective is not yet medical diagnosis or clinical finding interpretation. Instead, the goal is to test whether the selected vision-language models can reliably understand visual structure, answer questions over visual evidence, follow structured prompts, and produce machine-readable outputs that can later be adapted to medical MMVQA.
 
The baseline will use three generic visual-question-answering datasets:

<b>a) DVQA</b>
  This model is used for low-level visual parsing and question answering over bar charts. It is useful for testing whether a model can read labels, compare values, and extract information from visual layouts. DVQA is not medical, but it is useful as an early test of visual-symbolic reasoning. <br>
<b>b) ChartQA</b>
  This is used for stronger visual and logical reasoning over charts. It includes both human-written and generated questions and requires models to combine visual features with underlying chart information. This helps test whether the model can answer questions that require reasoning, not only visual recognition.

### Models to test

The model list should be divided into two categories.

#### A. Generic visual-reasoning and chart-specialist models

These models are useful for the generic baseline, but they are not medical models:

<b>Pix2Struct</b> — useful for screenshot-like visual-language tasks and structured visual layouts. It makes sense for Phase 1 because it was designed for visually situated language and can be fine-tuned on tasks involving screenshots, diagrams, tables, and visual layouts.<br>
<b>DePlot / MatCha / UniChart</b> — useful for chart understanding and chart-to-table or chart-reasoning tasks. These models are relevant for DVQA and ChartQA, but they should not be treated as strong candidates for capsule-endoscopy image interpretation. Their role is to establish chart and visual-structure baselines, not to identify lesions.

#### B. General multimodal models with possible medical transfer value

These models make more sense as candidates for the later medical MMVQA system:

<b>Qwen2.5-VL</b> — strong candidate because it supports image and video understanding, visual grounding, object localization, document/chart understanding, and long-video comprehension. These capabilities are relevant to capsule endoscopy because the final project needs frame-level reasoning, temporal context, lesion localization, and evidence grounding.<br>
<b>InternVL 2.5</b> — strong general multimodal baseline because it is an open multimodal large language model family designed for broad visual-language reasoning. It can be used as a competitive general VLM baseline before medical adaptation.<br>
<b>LLaVA / LLaVA-style models</b> — useful as open-source multimodal baselines because LLaVA connects a vision encoder with a language model and was designed for general-purpose visual instruction following.<br>
<b>LLaVA-Med</b> — should be added as a medical comparison model because it was specifically adapted for biomedical image conversation and medical VQA. It is more medically relevant than vanilla LLaVA and can help show whether a biomedical-adapted model performs better than a general VLM on medical visual questions.<br>
<b>Med-Flamingo</b> was designed for few-shot medical visual question answering.
 
#### Phase 2 — Build the raw medical data lake



#### Phase 3 — Turn raw data into visuals



    
#### Phase 4 — Construct the medical-QA benchmark


  
#### Phase 5 — Build the model stack


  
     
#### Phase 6 — Add retrieval and multimodal grounding





#### Phase 7 — Diagnosis engine


  

#### Phase 8 — Guarded remediation recommendation

 

## Implementation

The proposal is to implement this project in Python but using a very Clojure-like Declarative and Data Oriented styles which are both modern ways of managing the project. This effectively means that Python code will account the below rules:

  - plain data objects for all pipeline state
  - functions mostly pure that allows data transformations
  - avoid deep OOP hierarchies
  - represent diagnosis/action logic as data tables and transformations
  - treat each phase as a data pipeline
