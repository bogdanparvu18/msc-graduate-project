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

### Phase 1 — Establish the generic VQA baseline on DVQA & ChartQA 


  This phase establishes the project’s initial multimodal reasoning baseline before moving into capsule-endoscopy data. The objective is not yet medical diagnosis or clinical finding interpretation. Instead, the goal is to test whether the selected vision-language models can reliably understand visual structure, answer questions over visual evidence, follow structured prompts, and produce machine-readable outputs that can later be adapted to medical MMVQA.
 
The baseline will use three generic visual-question-answering datasets:

<b>a) DVQA</b>
  This model is used for low-level visual parsing and question answering over bar charts. It is useful for testing whether a model can read labels, compare values, and extract information from visual layouts. DVQA is not medical, but it is useful as an early test of visual-symbolic reasoning. <br>
<b>b) ChartQA</b>
  This is used for stronger visual and logical reasoning over charts. It includes both human-written and generated questions and requires models to combine visual features with underlying chart information. This helps test whether the model can answer questions that require reasoning, not only visual recognition.

### Models to test 
-----
The model list should be divided into two categories.

#### A. Generic visual-reasoning and chart-specialist models

These models are useful for the generic baseline, but they are not medical models:


- <b>Pix2Struct</b> — useful for screenshot-like visual-language tasks and structured visual layouts. It makes sense for Phase 1 because it was designed for visually situated language and can be fine-tuned on tasks involving screenshots, diagrams, tables, and visual layouts.<br>
- <b>DePlot / MatCha / UniChart</b> — useful for chart understanding and chart-to-table or chart-reasoning tasks. These models are relevant for DVQA and ChartQA, but they should not be treated as strong candidates for capsule-endoscopy image interpretation. Their role is to establish chart and visual-structure baselines, not to identify lesions.

#### B. General multimodal models with possible medical transfer value

These models make more sense as candidates for the later medical MMVQA system:

- <b>Qwen2.5-VL</b> — strong candidate because it supports image and video understanding, visual grounding, object localization, document/chart understanding, and long-video comprehension. These capabilities are relevant to capsule endoscopy because the final project needs frame-level reasoning, temporal context, lesion localization, and evidence grounding.<br>
- <b>InternVL 2.5</b> — strong general multimodal baseline because it is an open multimodal large language model family designed for broad visual-language reasoning. It can be used as a competitive general VLM baseline before medical adaptation.<br>
- <b>LLaVA / LLaVA-style models</b> — useful as open-source multimodal baselines because LLaVA connects a vision encoder with a language model and was designed for general-purpose visual instruction following.<br>
- <b>LLaVA-Med</b> — should be added as a medical comparison model because it was specifically adapted for biomedical image conversation and medical VQA. It is more medically relevant than vanilla LLaVA and can help show whether a biomedical-adapted model performs better than a general VLM on medical visual questions.<br>
- <b>Med-Flamingo</b> was designed for few-shot medical visual question answering.
 
### Phase 2 — Build the raw medical data lake

   It's also called data-engineering and data-curation phase as we build a structured raw medical data lake from capsule-endoscopy data before creating visual artifacts or question-answer pairs. The primary dataset will be Kvasir-Capsule, which contains capsule-endoscopy videos, labelled images, medically verified finding classes, bounding-box annotations, video identifiers, and frame numbers. The purpose of this phase is to transform the original dataset files into a clean, searchable, and reproducible project data layer.

The main objectives would be:

   - download Kvasir-Capsule
   - organize images, videos, metadata, labels, and bounding boxes
   - extract frames from videos
   - clean file paths and IDs
   - map each frame to its label and video source
   - normalize labels into broader clinical categories
   - create the structured foundation for later QA generation

   Kvasir-Capsule provides 47,238 medically verified labelled frames from 14 classes, together with 117 videos from which millions of frames can be extracted. The labelled images include class labels and bounding-box metadata, while the videos provide temporal context around the labelled findings. However, the full videos are not completely frame-by-frame annotated; many frames remain unlabelled. Therefore, this phase will explicitly separate labelled frames, partially labelled video context, and unlabelled video data.

The work in this phase includes downloading and registering the dataset, parsing the metadata file, linking each labelled image to its original video and frame number, extracting neighbouring frames around each labelled finding, normalizing medical labels into broader clinical categories, creating derived quality-control fields, and preparing video-level train, validation, and test splits. Data normalization step is required because medical datasets are often imbalanced. Some classes have many images, while rare findings have very few examples, the balance matter is also mentioned in the paper description of the dataset which creates a challenge for machine learning.
Since Kvasir-Capsule does not provide a complete good/bad image-quality label, the project will derive basic quality indicators such as reduced mucosal visibility, blur, brightness, contrast, and evidence sufficiency. If needed, an additional cleanliness or visibility dataset may be added later to strengthen this quality-control component.

The output of this phase is a curated raw medical evidence layer containing capsule frames, temporal context, labels, bounding boxes, metadata, quality indicators, and reproducible splits. This structured data lake becomes the foundation for Phase 3, where raw capsule-endoscopy data will be transformed into medical visual artifacts such as target frames, adjacent-frame strips, lesion crops, bounding-box overlays, and multi-panel clinical evidence packs. Clinical notes are not available in Kvasir-Capsule being removed by the authors, so any textual clinical context must be later generated using a pre trained LLM.

### Phase 3 — Turn raw data into visuals

The structured data collected in Phase 2 is transformed into reproducible medical visual artifacts that can later serve as inputs for multimodal question answering.

The purpose of this phase is to move from raw capsule-endoscopy material — such as videos, labelled frames, bounding boxes, and metadata — to a set of standardized visual representations that are easier for a vision-language model to interpret and reason over.

Unlike a conventional image-classification pipeline, this project cannot not rely only on isolated frames. Domain specific data as from capsule endoscopy is inherently temporal and context-dependent: a finding may become clearer over adjacent frames, may appear only briefly, or may need comparison with surrounding tissue to distinguish abnormal from normal patterns. For this reason, the system should not only preserve the target frame, but also generate contextual visual artifacts that capture neighboring frames, local lesion regions, visual progression, and image quality conditions.

The output of this phase will be a library of medical visual artifacts that can be paired later with natural-language questions in Phase 4.

The main objectives of this phase are:

- Standardize raw visual inputs into consistent artifact formats.
- Preserve temporal context around each labelled finding.
- Highlight diagnostically relevant regions using crops and overlays.
- Create richer multimodal inputs by combining visual evidence with structured metadata.
- Support multiple downstream QA task types, including detection, localization, temporal reasoning, comparison, and quality assessment.

Capsule endoscopy model construction differs from standard medical image classification in several ways:

- It is temporal
- A lesion may be visible more clearly in neighboring frames than in the labelled frame alone.
- It is context-sensitive
- A cropped lesion, a full-frame view, and a temporal strip may each provide different information.
- It contains quality variability
- Poor visibility, motion blur, bubbles, and debris may strongly affect interpretation.
- It benefits from structured multimodal input
- Metadata, labels, and visual annotations can be integrated with the image itself.

By transforming raw data into well-defined medical visual artifacts, Phase 3 provides the bridge between raw dataset preparation and the actual construction of the MMVQA benchmark.
    
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
