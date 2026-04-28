# Multimodal Question Answering over Network Visual Analytics with Safe Closed-Loop Remediation

## Executive workflow summary 

An operator asks a natural-language question about a network incident, and the system retrieves and reasons over dashboards, logs, topology views, routing evidence, and telemetry to produce an evidence-grounded diagnosis and a safe remediation recommendation.

The system will perform the following: 

1. Understand the operator’s natural-language question
2. Retrieve the relevant evidence \
  2a. use existing chart/dashboard VQA capability \
  2b. adapt it to network visuals 
3. Fuse visual and structured evidence \
  3a. dashboards, topology views, routing views \
  3b. telemetry, logs, config/routing context
4. Reason over the evidence
5. Return a grounded answer with confidence and evidence
6. Convert the answer into a diagnosis hypothesis
7. Map the diagnosis to candidate remediation actions
8. Apply safety checks before recommendation or execution \
  8a. confidence threshold \
  8b. policy/guardrails \
  8c. approval if needed \
  8d. post-change verification


## Problem statement
Modern network control-plane incidents are difficult to diagnose because the evidence is spread across multiple sources and formats: dashboards, topology views, Grafana panels, logs, routing events, metrics, and configuration changes. When BGP, IGP, BFD, interface events, or routing-policy changes create instability, operators usually identify symptoms first, then manually correlate data across tools to understand the cause, scope, and safest response. This process is slow, cognitively demanding, and often dependent on individual operator experience. As a result, mean time to diagnosis remains high, remediation decisions may be inconsistent, and operational risk increases when actions are taken from incomplete or weakly correlated evidence.

At the same time, existing question-answering systems and monitoring tools do not adequately support this workflow. Visual analytics platforms can display rich operational information, but they do not reason over it in a grounded way, and generic language models may summarize observations without enough operational context, evidence linkage, or action safety. Public benchmarks for chart and dashboard question answering show that multimodal reasoning over visual analytics is feasible, but network-specific question answering over control-plane evidence remains underdeveloped. In practice, network operators need a system that can answer natural-language questions such as: \
<i>

- What changed first? Which peer became unstable?
- Did route churn begin before interface errors? 
- What is the most likely cause? 
- What is the safest first action?</i>

This project work addresses that gap by proposing a multimodal question-answering system for network visual analytics that accepts natural-language operator questions, retrieves and reasons over dashboards, topology views, routing evidence, logs, and telemetry, and produces grounded answers with supporting evidence, diagnosis hypotheses, and safe remediation recommendations. The goal is not only to improve diagnosis speed, but also to make remediation more consistent and trustworthy through evidence grounding, confidence scoring, risk-aware action selection, approval gates, and constrained playbooks. The central research question is whether such a system can reduce mean time to diagnosis and support safer control-plane remediation without sacrificing operator trust and operational safety.

## Project workflow


####  Phase 0 — Lock the scope and incident taxonomy
  
  Below is a list of network incidents that can be later narrowed down.
  
  - BGP session clear / churn / withdrawal burst
  - route-policy regression
  - interface flap / admin shut / transceiver event
  - loss of alternate path / reduced path diversity
  - control-plane instability correlated with link or device events
 
    [Cisco’s public telemetry](https://github.com/cisco-ie/telemetry) repository already includes anomaly cases such as BGP Clear, Port Flap, Port Admin Shut, and Port Transceiver Pull and Reinsert, plus case files and topology documents. Routing-anomaly literature also gives us event types like outages, route leaks, and anomalous path behavior. 



####  Phase 1 — Establish the generic VQA baseline

  I plan to use two datasets, DVQA for bar-chart reading and ChartQA next for visual + logical chart reasoning. DVQA is specifically for question answering over bar charts, and ChartQA is stronger for reasoning because it includes both human-written and generated questions and explicitly combines visual chart features with the chart’s underlying data table. Also DashboardQA for dashboard specific analysis.
Models to run on these: Pix2Struct for screenshot-like visual language tasks, DePlot / MatCha / UniChart for chart-specific understanding and chart-to-table style reasoning and general multimodal baseline: Qwen2.5-VL, InternVL 2.5, LLaVA. 
  
#### Phase 2 — Build the raw network data lake

  At this stage I need to collect Telemetry / anomaly / KPI data from Cisco telemetry repo, Microsoft Cloud Monitoring Dataset, Exathlon (for anomaly detection),IBM Cloud Console dataset and NAB. Also get graphs from Internet Topology Zoo, CAIDA ITDK / topology collections, RIPE RIS raw dataset, BGPStream, BGPlay / RIPEstat BGPlay API,  additional telemetry datasets and/or lab-collected streaming telemetry via gNMI/OpenConfig (need to assert the feasibility).


#### Phase 3 — Turn raw data into visuals

  In this step the plan is to convert the raw network data into:
  
  - Grafana dashboard screenshots or panel images
  - Prometheus query graphs
  - topology images
  - routing-evolution views
  - multi-panel incident dashboards combining metrics, state timelines, and alerts.

  Phase 3 converts raw telemetry and routing data into reproducible visual analytics artifacts, including Grafana panels, Prometheus-derived metric graphs, topology views, and routing-evolution visualizations, which then serve as the visual input for the multimodal QA benchmark.

    
#### Phase 4 — Construct the network-QA benchmark

  At this stage I build the dataset schema + annotation pipeline. For each incident sample, create:

    visual inputs
    optional structured inputs
    question
    gold answer
    evidence
    diagnosis label
    remediation label
    risk label

  A JSON example is [here](dataset_schema.json). The gold answers must come from the underlying incident data and labels, not from the LLM to avoid hallucinacions. A gold answer will be grounded in incident truth data, supporting evidence references, a diagnosis label, and a remediation label. Natural-language QA are to be manually authored for a small subset of data, and LLM-assisted, but the source of truth will always come from the underlying incident data and annotations rather than from the LLM itself. \
  The incident truth object will be constructed by combining existing public labels and event metadata with derived facts extracted from telemetry, routing, topology, and visualization artifacts; public datasets provide partial ground truth, while diagnosis, evidence alignment, and remediation labels will be added through rule-based extraction and targeted human annotation.
  
#### Phase 5 — Build the model stack

  This phase builds the model stack by adapting a chart-specialist model for network dashboard understanding and one or two general multimodal vision-language models for end-to-end question answering over dashboards, topology views, routing visuals, and structured operational context, with structured outputs for evidence, diagnosis, and remediation recommendation.

   **Build the training dataset loaders** 
  
  This will read Phase 4 benchmark files and will create train/val/test loaders. I define few generic tasks:
  
   - open the sample JSON
   - load the referenced images
   - load structured text blocks
   - format everything into one prompt
   - return the gold JSON target

  The training will be performed to return 4 types of output: Answer generation, Evidence prediction, Diagnosis prediction, Action recommandation.
  
  **Prompt formatter**

  Each model family needs an exact prompt.

  **Fine-tuning pipeline**

  Models will be trained and adapted with the network benchmark information from the datasets. The base models still keep it generic pretraining knowledge and datasets teach it network vocabulary, incident patterns, output schema, and evidence habits. In this case fine-tuning data does not replace pretraining, it just aligns the model to networking domain in scope.

  **Structured output parser**

  A generated JSON needs to be validated.

  **Evaluation harness**

  A scoring system to be created: 

  - answer correctness
  - evidence correctness
  - diagnosis accuracy
  - action accuracy
     
#### Phase 6 — Add retrieval and multimodal grounding

  The runtime intelligence layer where the model does not look at all dashboards, all logs, and all routing history at once. In this section I implement a retrieval-and-grounding layer that maps operator questions to relevant multimodal evidence by combining metadata filtering and semantic retrieval over dashboard panels, topology views, routing artifacts, telemetry summaries, and logs. The selected evidence is then packaged into a constrained multimodal prompt so that the model produces answers, evidence references, and diagnosis candidates grounded (that's based on the actual evidence) only in retrieved incident artifacts.

  Subworkflows: operator question → retrieve the relevant evidence → package only that evidence → run the multimodal model → return answer + evidence + confidence


Within phase 6 of the project the system should do five things:

  - understand the natural-language question
  - retrieve the right multimodal evidence
  - format the evidence into a model-ready input pack
  - make the model answer only from that evidence
  - return structured evidence references with the answer

An example of input pack at this layer is exemplified [here](input_pack_p6.json). For the output of this phase the model should return a JSON like [this](output_p6.json).

#### Phase 7 — Diagnosis engine

  This is a typed inference layer that produces a diagnosis object which contains: diagnosis classes, extracted evidences, signal extraction, candidate diagnoses and scoring for it. I also add an uncertainty logic into the model to enhance the trust and safety and reduce the overconfidence. In this section the diagnosis engine that converts grounded multimodal evidence and question-answering outputs into structured incident diagnoses, confidence scores, affected scope, and candidate remediation paths using advanced reasoning.
  At this time of the project I propose a JSON object similar with [this one](output_p7.json) as the output of the model.
  

#### Phase 8 — Guarded remediation recommendation

  Phase 8 responds to the operator question on "What's the next action after the issue has been detected" and implements a guarded remediation layer that maps structured diagnoses to approved runbooks such as action playbooks, ranks candidate remediation steps by confidence and risk, applies policy-based safety checks and approval rules, and outputs constrained, operator-ready recommendations with required preconditions, post-checks, and rollback guidance before any execution. Depending on the stage of the project at the time of the presentation, part of this section might remain for future research and implementation.
  The plan is to have the input from Phase 7 and produce a confident output that contains proposed resolution actions that an operator should take to resolve the detected problem. In networking area, before recommending anything, an action should be tested whether it is even valid, what is the effective risk of implementing it and the blast radius (dimension of the potential impact). In real world applications, a policy risk/action table should result from this section after implementation:

  - medium risk + production edge → human approval required
  - high risk + low confidence → blocked
  - multi-cause uncertain → operator review only
  - very low risk evidence-collection action → can be auto-executed (later stage/other automation triggered)

## Implementation

The proposal is to implement this project in Python but using a very Clojure-like Declarative and Data Oriented styles which are both modern ways of managing the project. This effectively means that Python code will account the below rules:

  - plain data objects for all pipeline state
  - functions mostly pure that allows data transformations
  - avoid deep OOP hierarchies
  - represent diagnosis/action logic as data tables and transformations
  - treat each phase as a data pipeline
