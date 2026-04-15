# Multimodal Question Answering over Network Visual Analytics with Safe Closed-Loop Remediation
A dedicated repository documenting the development, research, and implementation of my Graduate Project in fulfillment of the Master of Science in Computer Science at Lakehead University.

## Executive workflow summary 
1. start from existing chart/dashboard VQA knowledge,
2. adapt it to network visuals,
3. add network telemetry, topologies and routing context,
4. output diagnosis and recommended actions,
5. validate actions before execution.

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

- Phase 0 — Lock the scope and incident taxonomy
  
  Below is a list of network incidents that can be later narrowed down.
  
  - BGP session clear / churn / withdrawal burst
  - route-policy regression
  - interface flap / admin shut / transceiver event
  - loss of alternate path / reduced path diversity
  - control-plane instability correlated with link or device events
 
    [Cisco’s public telemetry](https://github.com/cisco-ie/telemetry) repository already includes anomaly cases such as BGP Clear, Port Flap, Port Admin Shut, and Port Transceiver Pull and Reinsert, plus case files and topology documents. Routing-anomaly literature also gives you event types like outages, route leaks, and anomalous path behavior.

- Phase 1 — Build the raw data lake
- Phase 2 — Turn raw data into visuals
- Phase 3 — Construct the network-VQA benchmark
- Phase 4 — Build the model stack
- Phase 5 — Add retrieval and multimodal grounding
- Phase 6 — Diagnosis engine
- Phase 7 — Guarded remediation recommendation
