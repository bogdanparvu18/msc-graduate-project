# Draft Ideas — Multimodal RAG Experiments for MMVQA

> **Status:** Research ideas / proposed future work — **not implemented**.  
> **Created:** 2026-10-23  
> **Target:** [Phase 5 — Add retrieval and multimodal grounding](README.md#phase-5--add-retrieval-and-multimodal-grounding), with evaluation in Phase 8.  
> **Dataset:** Kvasir-Capsule. These ideas should be refined after the dataset, evidence-unit definitions, and final train/validation/test split are fixed.

## 1. Motivation and research question

A key question for capsule-endoscopy retrieval-augmented generation (RAG) is **which information should form the retrieval query**:

- The clinician's natural-language question;
- A representative capsule-endoscopy frame;
- A joint representation of the question and the representative frame.

Text-only retrieval can capture the *intent* of the question but may miss its visual context. Image-only retrieval can capture visual similarity but may retrieve findings unrelated to the specific question. A multimodal query could combine both signals to retrieve more relevant evidence.

**Main research question:** Does joint image–question retrieval improve evidence relevance, grounding, and downstream medical VQA performance compared with text-only and frame-only retrieval, under controlled experimental conditions?

**Secondary question:** When does adding adjacent-frame temporal context help, especially for short-lived, ambiguous, or recurring findings?

These are hypotheses to be tested, not established conclusions for Kvasir-Capsule.

## 2. Proposed experimental configurations

| ID | Retrieval strategy | Query representation | Purpose |
| --- | --- | --- | --- |
| **No-RAG** | No retrieved evidence | Question + designated visual input to the generator | Establish a generator-only baseline. |
| **RAG-T** | Text-only | Embedding of the clinical question | Measure retrieval driven by question semantics. |
| **RAG-V** | Visual-only | Embedding of the representative frame | Measure retrieval driven by image content alone. |
| **RAG-MM** | Joint multimodal | Fused question and representative-frame embeddings | Test whether question-aware visual retrieval improves relevance. |
| **RAG-MM-GMU** | Learned multimodal fusion | Gated Multimodal Unit (GMU), trained using training-only data | Test adaptive weighting of text and image features against a simpler fusion baseline. |
| **RAG-MM-T** *(optional extension)* | Multimodal + temporal | Question, representative frame, and selected neighboring frames or temporal summary | Quantify the additional value of temporal context. |

The **three primary RAG baselines are RAG-T, RAG-V, and RAG-MM**. No-RAG is a generation control; GMU and temporal context are subsequent ablations.

An initial RAG-MM baseline can use normalized, dimension-compatible embeddings with fixed weighted fusion or a learned projection. Compare it with GMU only after establishing a reproducible baseline. If the text and visual encoders use different vector spaces, do **not** add raw vectors without an appropriate alignment/projection mechanism.

### Key hypothesis

RAG-MM may perform better for questions in which **visual evidence and question semantics are both necessary**; RAG-T may remain competitive for terminology-driven queries, and RAG-V may perform well for visually distinctive findings. The expected outcome should be measured by question category rather than assuming a single method wins universally.

## 3. Evidence corpus and integration with the existing pipeline

Reuse existing project artifacts rather than rebuilding the dataset:

- **Phase 2, Sector 5:** Finding segments and frame/annotation identities.
- **Phase 2, Sector 6:** Provenance-tracked representative and neighboring frames, temporal references, and source-video/frame identifiers.
- **Phase 2, Sector 7:** Image quality-control indicators and source integrity.
- **Phase 2, Sector 8:** Standardized visual artifacts, such as crops, overlays, temporal collages and multi-panel evidence, with source provenance and ordered input recipes.
- **Phase 3:** QA examples, question types, expected answers, and evidence references.

Build a **canonical evidence catalog** with stable identifiers and provenance, then maintain the modality-specific index views required by each retriever. Candidate evidence may include verified frames, finding segments, visual derivatives, label definitions, curated terminology, dataset documentation, and appropriately sourced external medical reference text.

**Kvasir-Capsule does not provide patient-specific clinical notes.** Do not present synthetic or reference text as original patient reports.

The initial experiments should explicitly distinguish:

1. **Query-local evidence retrieval:** locating relevant frames or segments within the examination or visual context associated with the question.
2. **Reference/knowledge retrieval:** finding relevant examples or curated external textual evidence.

These tasks have different leakage risks and notions of relevance; do not combine their scores without clear definitions. In particular, avoid a trivial image-retrieval test in which the exact query image is indexed and counted as a successful retrieved result.

### Provenance fields to preserve

Each result should be traceable to the source video, verified frame index, physical-frame identity, finding/segment identifier where available, artifact identifier, quality-control status, retrieval score, and index/model version. If a field is not available, record it as missing rather than inventing it.

## 4. Experimental protocol and leakage controls

1. **Finalize the dataset split before model fitting or final evaluation.** Sector 6 intentionally deferred final split assignment. Resolve frame/segment identities and split isolation before using the artifacts for training or evaluation.
2. **Use group-aware isolation.** Keep the same physical frame, its derivatives, and overlapping temporal windows from leaking between train, validation, and test. Evaluate whether stricter video-level isolation is feasible; document any same-video cross-split exposure explicitly.
3. **Fit fusion modules on training data only.** Never tune GMU gates, retrieval weights, thresholds, or rerankers using the test partition.
4. **Control the comparison.** Keep the evidence catalog, top-K budget, downstream generator, prompting, output schema, and inference hardware fixed where feasible. Log necessary differences between text, visual, and joint embedding/index implementations.
5. **Compare like-for-like.** Maintain the same set of question IDs, relevant evidence annotations, and clinical categories across configurations.
6. **Prevent answer leakage.** Retrieval documents must not disclose test ground-truth answers, withheld annotations, or generated QA targets. Clearly distinguish permissible same-case source evidence from prohibited training/test contamination.
7. **Evaluate both matched and difficult cases.** Include visually similar findings, low-visibility images, repeated findings, short temporal events, and cases where the evidence is genuinely insufficient.
8. **Report cost as well as quality.** Track index size, indexing time, retrieval latency, reranking latency (if used), generation time, tokens, and GPU memory.

Unlabelled temporal frames must **not** be treated as confirmed negative or normal examples solely because they have no annotation.

## 5. Evaluation: retrieval is not answer correctness

Use three separate evaluation layers.

| Layer | Candidate metrics | What it tests |
| --- | --- | --- |
| **Retrieval** | Recall@K, Precision@K, MRR, nDCG@K; segment/evidence recall | Whether the right evidence was recovered and ranked well. |
| **Grounding** | Evidence precision/recall/F1, correct source citation, localization IoU, temporal overlap/recall | Whether generated claims are supported by the correct frame, region, or temporal span. |
| **Answer and safety** | Answer accuracy/F1 as appropriate, finding/category accuracy, unsupported-claim rate, abstention accuracy, JSON validity, review-label agreement | Whether the answer is correct, appropriately qualified, and safe. |

Evaluate retrieval and generation **jointly and independently**. An improved Recall@K does not prove improved VQA accuracy; a correct answer does not prove that the model used or cited the correct evidence.

When clinically equivalent evidence appears in multiple frames of the same finding segment, define relevance sets that can recognize more than one valid evidence reference. Do not relax split isolation or accept unsupported evidence to inflate these metrics.

Use paired comparisons on the same questions; report per-category results, uncertainty intervals where practical, and representative failure cases. Separate *no evidence*, *poor quality*, and *conflicting evidence* scenarios.

## 6. Optional extensions (after the primary ablation)

- **Temporal retrieval:** Compare a representative frame with a short frame window or temporal artifact built from Sector 6. Control the number of frames/tokens so any gain is interpretable.
- **Multimodal reranking:** After first-stage retrieval, rerank a fixed candidate set using both the image and question. Report quality–latency trade-offs separately from first-stage retrieval.
- **Adaptive retrieval:** Test whether some questions benefit from text, vision, or multimodal retrieval chosen dynamically. Compare against always-on RAG-MM.
- **Evidence-aware abstention:** Require the generator to abstain when retrieved evidence is absent, low-quality, conflicting, or insufficient. Keep the final clinical review recommendation guarded and non-autonomous.
- **Controlled knowledge grounding:** Add curated terminology, clinical concept relationships, or knowledge-graph references as an additional factor *after* the three retrieval baselines have been established.

## 7. Proposed implementation milestones

- [ ] Freeze a versioned, split-safe evidence catalog and define its retrieval units.
- [ ] Specify question-level and finding-segment-level relevance annotations.
- [ ] Implement and evaluate **RAG-T** and **RAG-V** first.
- [ ] Implement **RAG-MM** with a reproducible simple fusion baseline.
- [ ] Train and ablate **RAG-MM-GMU** using training-only supervision.
- [ ] Add optional **RAG-MM-T** using Sector 6 context after confirming temporal isolation.
- [ ] Run the same frozen VLM generator and structured-output validator on all configurations.
- [ ] Produce a single comparison table covering retrieval, grounding, VQA quality, safety, and runtime costs.
- [ ] Publish error analyses, exact configurations, evidence manifests, model/index versions, and seeds.

**Completion criterion:** An experiment is considered comparable only when its split assignment, candidate corpus, evidence relevance definitions, generator configuration, and evaluation artifacts are recorded and reproducible.

## 8. Research reference and status

**Primary inspiration:** Daniela Jimenez-Pano, Hugo Jair Escalante, and Delia Irazú Hernández-Farías, *Enhancing RAG-based Medical VQA with Gated Multimodal Representations*, arXiv:2610.23588, submitted 16 October 2026. [Paper](https://arxiv.org/abs/2610.23588).

The paper studies GMU-based fusion of medical image and question representations for retrieval while keeping the underlying medical vision-language generator frozen. It is a **preprint submitted to ECIR**, not an established result on capsule-endoscopy data. Its reported findings are motivation for an ablation, **not** evidence that GMU will improve Kvasir-Capsule performance.

**Project placement:** This draft augments the existing [Phase 5 retrieval and multimodal grounding plan](README.md#phase-5--add-retrieval-and-multimodal-grounding). It does not replace the current Phase 2 sector implementations or claim that RAG experiments have been run.
