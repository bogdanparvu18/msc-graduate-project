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

---

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

---

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

---

### Phase 4 — Construct the medical-QA benchmark

In this phase, the visual artifacts created in Phase 3 are transformed into a structured medical multimodal visual question-answering benchmark.

The goal is to create a dataset where each sample contains:

- a visual input, such as an image, crop, frame strip, video clip, or multi-panel artifact
- a natural-language clinical-style question
- a ground-truth answer
- evidence grounding
- task type
- source label
- normalized clinical category
- bounding-box reference
- temporal context
- retrieved medical context

This phase converts Kvasir-Capsule from a labelled medical visual dataset into a medical MMVQA benchmark.

Kvasir-Capsule provides the visual ground truth:

   - source video
   - target frame
   - original label
   - bounding box
   - frame number
   - labelled/unlabelled status
   - anatomical or luminal finding class

The LLM provides the language layer:

- question wording
- answer phrasing
- explanation templates
- evidence statements
- uncertainty/abstention wording

The KG/RAG system provides the clinical context layer:

- terminology mapping
- label definitions
- relationships between findings and categories
- clinical relevance of findings
- guideline-style background where appropriate

The benchmark will be constructed using an LLM-assisted QA generation pipeline. The LLM will generate clinical-style questions, answers, explanations, and uncertainty statements, but it will not create independent medical ground truth. All generated QA pairs will be grounded in Kvasir-Capsule annotations, including the original label, bounding box, source video, frame number, and normalized clinical category.

A knowledge-grounding layer will also be introduced in this phase. Since Kvasir-Capsule does not provide real clinical notes, this layer will use external curated medical texts, terminology resources, dataset documentation, and guideline-style references rather than patient-specific notes. A medical knowledge graph will represent relationships between findings, anatomical landmarks, broad clinical categories, visual evidence types, quality limitations, and supported question types. A vector database will store semantic embeddings of relevant unstructured medical text snippets. Hybrid retrieval will combine semantic search from the vector database with graph traversal from the knowledge graph to provide controlled medical context to the LLM. The LLM will then generate evidence-grounded QA pairs using this retrieved context together with the dataset annotations.

The benchmark will include several question categories: visual identification questions, evidence-grounding questions, temporal-context questions, localization questions, normal-versus-abnormal questions, uncertainty or abstention questions, and clinical review prioritization questions. These question types will allow the benchmark to evaluate not only whether a model can recognize a finding, but also whether it can localize evidence, use temporal context, distinguish normal from abnormal views, recognize poor image quality, and abstain when the visual evidence is insufficient.

The output of Phase 4 will be a validated MMVQA benchmark split into training, validation, and test sets. The final dataset will include artifact paths, questions, answers, evidence fields, task labels, difficulty levels, KG concepts, retrieved context references, and validation status. This phase forms the bridge between medical visual artifact generation and the later model training and evaluation phases.

---

### Phase 5 — Build the medical model stack

In this phase, the project builds the medical multimodal model stack that will be trained, adapted, prompted, evaluated, and compared using the medical MMVQA benchmark created in Phase 4.

The model stack should not be limited to a single model. It should include:

- general multimodal vision-language models-
- medical-adapted multimodal models
- optional region-aware or grounding-aware model components
- a structured output layer
- a medical evidence-grounding evaluator
- a safety and uncertainty layer
- a RAG-ready input/output interface for later integration in Phase 6

The goal is not to build a final clinical diagnosis system. The goal is to build a research-grade medical MMVQA system that can answer questions about capsule-endoscopy visual artifacts, cite available visual evidence, classify possible findings, localize abnormalities, estimate uncertainty, and flag cases for clinician review.

The model created in this phase is not trained to make final clinical decisions. It is trained to answer medical visual questions, ground its answers in the available benchmark evidence, identify possible findings, and support clinician review. The output of the MMVQA system will be formalized in a structured JSON-like format similar to [this example](output_p5.json).


The model stack will include several components.

First, general multimodal vision-language models such as **Qwen2.5-VL, InternVL, and LLaVA-style models** will be used as baseline systems for image-question answering, multi-image reasoning, visual grounding, and structured answer generation.

Second, medical-adapted multimodal models such as **LLaVA-Med and Med-Flamingo** will be used as medical comparison baselines. These models will help evaluate whether biomedical adaptation improves performance over general-purpose VLMs on capsule-endoscopy visual question answering.

Third, chart/dashboard-specialist models from Phase 1, such as **Pix2Struct, DePlot, MatCha, and UniChart**, will remain baseline-support components only. They are useful for generic visual-structure reasoning and for maintaining continuity with the original dashboard-based VQA plan, but they are not expected to perform strong capsule-endoscopy interpretation.

The model stack will be trained and evaluated on multiple input types, including single frames, lesion crops, bounding-box overlays, frame-strip timelines, finding progression panels, quality-control panels, normal-vs-abnormal comparison panels, and multi-panel medical artifacts.

Each model will receive a visual artifact and a natural-language question. In this phase, the model may also receive structured benchmark fields already attached to the sample, such as artifact type, available evidence references, bounding-box availability, temporal-context availability, quality tag, source label, or normalized clinical category.

However, Phase 5 will not yet implement dynamic retrieval from a vector database, knowledge graph, or artifact index. Full retrieval, RAG, and multimodal grounding will be introduced in Phase 6.

The expected output will be structured JSON rather than free text. Each output should include the answer, clinical finding, broad clinical category, evidence references, localization information, confidence score, uncertainty reason, decision-support label, and safety flags. This structured format allows the system to be automatically validated and evaluated for answer correctness, evidence grounding, localization accuracy, temporal reasoning, uncertainty handling, JSON validity, and clinical safety.

Training will proceed in stages.

At the first stage, candidate models will be evaluated zero-shot on the Phase 4 benchmark. This will establish baseline performance before adaptation.

Second, prompt-engineered baselines will be tested to improve JSON compliance, reduce hallucination, and enforce safer medical-answering behavior.

Third, selected open models will be fine-tuned using **LoRA** or **QLoRA** on the benchmark training split. The fine-tuning objective will be multi-task and will include answer generation, evidence prediction, clinical finding classification, broad category classification, localization, temporal-context reasoning, uncertainty estimation, and decision-support label prediction.

The evaluation harness will measure several dimensions: answer accuracy, fine-grained clinical label accuracy, broad category accuracy, evidence-reference correctness, localization accuracy, temporal reasoning accuracy, uncertainty and abstention accuracy, JSON validity, schema compliance, and clinical safety.

The model will be penalized for unsupported medical claims, invented patient history, invented symptoms, treatment recommendations, prognosis claims, or final clinical decisions. The safest allowed decision-support output will be a label such as **“flag_for_clinician_review”**, not a diagnosis or treatment recommendation.

The final output of Phase 5 will be a tested medical MMVQA model stack, including baseline comparisons, medical model comparisons, prompt-engineered baselines, fine-tuned model variants, structured output validation, evidence-reference prediction, localization evaluation, uncertainty and abstention evaluation, clinical safety checks, and ablation studies comparing different visual artifact types.

Phase 5 will produce a **RAG-ready medical MMVQA model**, but not the full retrieval-grounded system. The full retrieval and multimodal grounding layer will be added in Phase 6, where the trained model will be connected to visual artifact retrieval, metadata retrieval, temporal retrieval, vector database retrieval, medical knowledge graph retrieval, hybrid evidence packaging, and grounding validation.

Below is a list of models, input type expected and the purpose of the experiment.

| Experiment | Model               | Input                                       | Purpose                                   |
| ---------- | ------------------- | ------------------------------------------- | ----------------------------------------- |
| E1         | LLaVA / general VLM | single frame                                | basic VQA baseline                        |
| E2         | Qwen2.5-VL          | frame + crop / overlay                      | strong general VLM baseline               |
| E3         | InternVL            | multi-panel artifact                        | general multimodal comparison             |
| E4         | LLaVA-Med           | single frame / crop                         | medical VLM baseline                      |
| E5         | Med-Flamingo        | few-shot medical examples                   | medical few-shot baseline                 |
| E6         | selected VLM        | Phase 4 QA benchmark                        | fine-tuned capsule-endoscopy model        |
| E7         | fine-tuned model    | structured evidence fields already attached | tests JSON output and evidence prediction |
| E8         | fine-tuned model    | poor-quality / uncertainty samples          | tests abstention and safety               |

---
     
### Phase 6 — Add retrieval and multimodal grounding

At this stage we add retrieval and grounding so the model can answer using the correct frame, crop, bounding box, temporal context, metadata, KG concepts, and retrieved clinical text — instead of producing unsupported free-text answers.

In this phase, the medical MMVQA system is extended with retrieval and grounding mechanisms so that model answers are based on explicit evidence rather than unsupported free-text generation. The system will retrieve relevant visual artifacts, metadata, temporal context, medical knowledge graph relations, and clinical text snippets before generating an answer.

The retrieval layer includes several components. A visual artifact index stores target frames, lesion crops, bounding-box overlays, frame-strip timelines, progression panels, quality-control panels, and multi-panel artifacts. A metadata retriever provides exact information such as original label, normalized clinical category, bounding-box coordinates, source video, frame number, and artifact type. A temporal retriever recovers neighboring frames or short clips around a target frame. A **vector database**  stores embeddings of curated clinical text, dataset documentation, label definitions, and medical terminology. A **knowledge graph** represents structured relationships between findings, broad categories, anatomical landmarks, artifact types, evidence types, question types, and safety rules.

The system uses hybrid retrieval to combine these sources. For example, a localization question should retrieve the target frame, bounding-box overlay, lesion crop, and bounding-box metadata. A temporal question should retrieve the frame-strip timeline and adjacent frames. A clinical category question should retrieve the original label, normalized category, relevant KG relations, and medical terminology snippets. This evidence is packaged and passed to the multimodal model together with the question.

The model then [outputs structured JSON](output_p6.json) that includes the answer, clinical finding, broad category, evidence references, localization field, confidence, uncertainty reason, and decision-support label. A grounding validator checks whether the answer is supported by the retrieved evidence. It will verify that cited evidence exists, that the predicted finding matches the metadata label, that the broad category is consistent with the label map and KG, that localization agrees with the bounding box where applicable, and that the model does not invent unsupported patient history, treatment recommendations, or final clinical diagnoses.

This phase also evaluates retrieval performance and grounding quality. Retrieval is measured using Recall@K, Precision@K, MRR, temporal recall, and evidence recall. Model grounding is measured using answer accuracy, evidence F1, localization IoU, temporal grounding accuracy, abstention accuracy, hallucination rate, safety violations, and JSON validity. Ablation experiments compare no retrieval, metadata-only retrieval, visual-only retrieval, text-only retrieval, KG-only retrieval, and full hybrid retrieval.

The final output of Phase 6 is a retrieval-grounded medical MMVQA pipeline that can answer capsule-endoscopy questions using explicit evidence from visual artifacts, metadata, temporal context, KG relations, and retrieved clinical text.

---

### Phase 7 — Clinical finding interpretation engine

The purpose of this phase is to build a post-model interpretation engine that receives the structured output from the medical MMVQA model and converts it into a safer, cleaner, finding-level interpretation. We don't train or fine tune the model again but at this stage it happens more reasoning, normalization, and interpretation layer after the model output. We produce a separated findin interpretation object of [this JSON format](output_p7.json). This object will not make a final clinical decision. Instead, it will provide a verified, conservative, evidence-aware interpretation that Phase 8 can use to assign a guarded decision-support label such as routine review, flag for clinician review, abstain due to poor visibility, or insufficient evidence.

The interpretation engine validates the structured model output and check that all required fields are present. It then normalizes the predicted finding to the allowed Kvasir-Capsule label set, validate the broad clinical category, verify evidence references, interpret localization, summarize temporal behavior, and assess uncertainty. It also detects contradictions, such as a finding-category mismatch, unsupported localization claim, or a confident answer despite poor image quality. The summary uses conservative language such as “model-supported visual finding” or “possible visual finding” and will explicitly avoid final diagnosis, treatment recommendation, prognosis, or patient-specific claims.

For temporal capsule-endoscopy cases, the engine may also group repeated detections across adjacent frames into a finding episode. This allows the system to describe whether a finding appears only in the target frame, persists across neighboring frames, becomes clearer over time, or is uncertain due to motion or poor visibility.
  
---

### Phase 8 — Guarded remediation recommendation

 

## Implementation

The proposal is to implement this project in Python but using a very Clojure-like Declarative and Data Oriented styles which are both modern ways of managing the project. This effectively means that Python code will account the below rules:

  - plain data objects for all pipeline state
  - functions mostly pure that allows data transformations
  - avoid deep OOP hierarchies
  - represent diagnosis/action logic as data tables and transformations
  - treat each phase as a data pipeline
