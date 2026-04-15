# Multimodal Question Answering over Network Visual Analytics with Safe Closed-Loop Remediation
A dedicated repository documenting the development, research, and implementation of my Graduate Project in fulfillment of the Master of Science in Computer Science at Lakehead University.

## Executive workflow summary 
1. start from existing chart/dashboard VQA knowledge,
2. adapt it to network visuals,
3. add network telemetry, topologies and routing context,
4. output diagnosis and recommended actions,
5. validate actions before execution.

## Problem statement
Modern control-plane incidents in large scale networks are difficult to diagnose and remediate because the evidence is fragmented across multiple modalities: telemetry dashboards, topology views, routing-event histories, alarm timelines, and device or policy changes. When BGP, IGP, BFD, or routing-policy changes introduce instability, operators often detect symptoms late, correlate these sources manually, and rely on brittle runbooks that are hard to verify before execution. This makes response slower, increases operational risk, and leaves remediation quality highly dependent on human experience rather than consistent, reproducible reasoning. At the same time, the visual artifacts operators use every day are not trivial screenshots; they are dense analytical views that encode time, state, correlation, and topology, and they increasingly resemble the kinds of chart- and dashboard-centered inputs already studied in modern VQA research. Benchmarks such as DVQA, ChartQA, and DashboardQA show that visual question answering has already expanded beyond natural images to bar charts, chart reasoning, and interactive dashboards, while also showing that these tasks remain challenging even for strong multimodal models. 

Despite that progress, there is still a major gap between general-purpose VQA and real network operations. **Existing VQA benchmarks** cover visual reasoning over charts and dashboards, but public datasets and benchmarks specifically targeting network control-plane question answering are still immature. In practice, operators need to ask questions such as: _Which peer changed state first? Which path disappeared after the policy push? Did route churn start before or after interface errors? What is the most likely cause of the instability?_ Answering these questions requires a system that can reason jointly over rendered visual evidence and structured network state, rather than over either one in isolation. **ChartQA** is especially relevant here because it explicitly combines visual features with the chart’s underlying data table, suggesting that a network-oriented system should likewise combine dashboard and topology visuals with telemetry and routing data. **DashboardQA** further shows that even current multimodal agents still struggle with grounding, interaction planning, and reasoning over real dashboards.

This project addresses that gap by proposing **a grounded multimodal VQA system for network control-plane incident diagnosis**. The system will ingest network dashboards, topology views, routing-evolution views, and structured telemetry or routing state, answer operator questions with evidence and confidence, convert those answers into incident diagnoses, and then map diagnoses to guarded remediation recommendations. Because automation in operations is increasingly judged not only by speed but by verifiability, safety, and human trust, the project does not stop at diagnosis. Candidate actions must be validated in a network digital twin and subjected to risk-based approval policies before execution. Current IETF work on network digital twins and agentic AI explicitly highlights reference architectures, validation workflows, and human-in-the-loop control for AI-driven network operations, which makes this a timely and technically relevant research direction.

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
