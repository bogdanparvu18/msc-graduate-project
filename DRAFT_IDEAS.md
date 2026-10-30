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

---

## 9. Candidate Medical Visual Tokenizer — MedVTok (review later)

> **Added:** 2026-10-27  
> **Status:** **Draft research idea / not implemented or benchmarked on Kvasir-Capsule.**  
> **Open-source availability:** Official [GitHub repository](https://github.com/Masaaki-75/medvtok) contains model, inference, and training code and states that original project contributions are under the **MIT license**. Its README links to a [pretrained checkpoint on Hugging Face](https://huggingface.co/massaki75/medvtok) (including `config.json` and `ckpt.pth`); actual checkpoint download and execution have **not yet been verified** for this project. Third-party components and expert models have their own licenses.

**Paper:** Chenglong Ma et al., *MedVTok: A General-Purpose Medical Visual Tokenizer*, NeurIPS 2026, [arXiv:2610.35842](https://arxiv.org/abs/2610.35842) (arXiv submission: 26 October 2026).  
**Code:** https://github.com/Masaaki-75/medvtok  
**Weights:** https://huggingface.co/massaki75/medvtok  
**Setup/inference guide:** https://github.com/Masaaki-75/medvtok/blob/main/GUIDELINE.md

### Why it is interesting

MedVTok maps 2D medical images and 3D medical volumes from heterogeneous imaging modalities into a shared visual token space. Its contributions include:

- **Slice Differential Regularization (SliDR):** encourages anatomical continuity across adjacent slices in volumetric data.
- **Multi-Expert Representation Alignment (MERA):** aligns outputs with complementary medical expert encoders while retaining modality-specific information.
- A shared encoder/decoder interface exposing features, spatial patch tokens, and latent representations for multiple tasks.

The authors report training on **28 million images and 70,000 volumes across nine imaging modalities**, and evaluation on **24 downstream benchmarks** (classification, retrieval, segmentation, synthesis, VQA, and reporting). These are author-reported outcomes, **not evidence of a performance gain on capsule endoscopy**.

### Proposed use in this MMVQA project

**Primary candidate:** Evaluate MedVTok as an alternative **visual feature extractor** in the planned Phase 5 RAG retrieval ablation, especially:

1. **RAG-V:** Compare embeddings/features from MedVTok on representative capsule frames with the current general-purpose visual retrieval encoder baseline (and a medical encoder baseline, if available).
2. **RAG-MM:** Combine question embeddings with MedVTok frame features using an explicitly aligned/projection-based fusion method; compare with the original text+visual fusion baseline. Do not assume MedVTok's visual embedding space is already aligned to arbitrary text embeddings.
3. **RAG-MM-GMU (optional):** Test whether a train-only learned gate improves retrieval relevance over fixed fusion for the same feature encoders.
4. **Temporal extension (optional):** Extract MedVTok representations independently per verified frame/window from Phase 2 Sector 6, then evaluate temporal pooling or aggregation. **Its 3D volume capability is not automatically video understanding**: anatomical slice continuity and time-ordered endoscopy frames are different problems.

**Key ablation:** Freeze the downstream generator, question set, evidence catalog, split definitions, top-K, and evaluation protocol; change only the visual encoder/tokenizer where possible. Compare **Recall@K, MRR, nDCG@K, evidence grounding, answer accuracy, small-lesion/localization fidelity, extraction/indexing latency, throughput, and GPU memory**. Report results by finding category and image-quality condition.

**Risks and checks:** Confirm (a) checkpoint availability and external expert dependencies, (b) inference compatibility and memory needs in Colab, (c) licensing of code, weights, and dependencies, (d) relevance of the pretraining imaging modalities to capsule endoscopy, and (e) whether spatial resolution/token compression loses subtle visual findings. Preserve the project's group-aware split and provenance controls to avoid evidence leakage.

### Follow-up checklist

- [ ] Download and verify the official MedVTok checkpoint and run the included 2D inference example.
- [ ] Inspect the MIT repository license and individual third-party/weight usage terms.
- [ ] Extract features for a small, **split-safe** sample of Kvasir-Capsule representative frames.
- [ ] Assess image resolution, fine-grained pathology preservation, and embedding separation across findings.
- [ ] Decide whether it warrants full RAG-V and RAG-MM benchmarking against existing encoders.
- [ ] Record measured results and limitations before considering integration into the project pipeline.

**Decision:** **Keep as an open-source encoder candidate to review later; do not treat it as an implemented Phase 5 component or a validated improvement.**

---

## 10. Candidate Unified Multimodal Embedder — EmbeddingGemma 2 (review later)

> **Added:** 2026-10-30  
> **Status:** Draft research idea — **not implemented, downloaded, or evaluated on Kvasir-Capsule**.  
> **Availability:** Google DeepMind provides an **open-weight model under Apache 2.0** on [Hugging Face](https://huggingface.co/google/embeddinggemma-2), with a documented inference API. The public model launch was **6 October 2026**; the detailed arXiv technical report was submitted on **27 October 2026**. These are different publication events.

**Paper:** Henrique Schechter Vera et al., *EmbeddingGemma 2: Unified and Efficient Multimodal Representations*, [arXiv:2610.37364](https://arxiv.org/abs/2610.37364).  
**Model/weights:** https://huggingface.co/google/embeddinggemma-2  
**Technical documentation:** https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2  
**Release announcement:** https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/

### Research motivation

EmbeddingGemma 2 projects **text, images, video, audio, and interleaved multimodal inputs** into a **shared 768-dimensional embedding space**. This is especially interesting for the planned **RAG-T, RAG-V, and RAG-MM** experiments: unlike a visual-only medical tokenizer, the same pretrained model can embed both natural-language clinical questions and representative endoscopy frames, including a combined question–image input.

Google reports approximately **740M total parameters** (744M in the technical paper), with modular text (~270M), vision (~170M), and audio (~300M) components. The text+vision configuration is roughly **440M parameters** and does not need the audio encoder. Matryoshka Representation Learning supports **768, 512, 256, or 128-dimensional embeddings**, potentially reducing vector-index size at a measurable retrieval-quality cost.

**Reported evidence (not MMVQA results):** In the authors' MMEB-v2 ablation, multimodal pre-finetuning scored **52.27**, additional roughly compute-matched distillation scored **52.28**, and the contrastive/spread-out finetuning stage scored **55.46** on the overall metric. Thus the improvement is attributable to the different finetuning objective rather than simply extending distillation in this experiment. These figures are from **Table 3 of the paper**, not a benchmark on Kvasir-Capsule. The complete released model's MMEB-v2 overall score is reported separately; do **not** confuse the ablation checkpoints with its final cross-task evaluation.

### Proposed MMVQA evaluation

Use this as a **candidate retrieval encoder**, not as a replacement for the response-generating VLM.

1. **RAG-T — text only:** Embed the medical question with the documented query prompt. Search a text index of permitted evidence descriptions, metadata, label definitions, and curated clinical references. Avoid including test answers or derived evaluation labels in the retrieval text.
2. **RAG-V — frame only:** Embed the verified representative capsule frame, retrieve visually relevant evidence from the same split-safe candidate catalog, and exclude exact query-frame/self-duplicates when the task is finding other evidence.
3. **RAG-MM — question + frame:** Embed an interleaved clinical question and its representative image as **one multimodal query**, then retrieve relevant evidence in the shared embedding space. Test whether this outperforms text-only and frame-only retrieval on the *same question IDs and relevant evidence judgments*. This is different from merely concatenating independently trained text and visual vectors.
4. **RAG-MM-T (optional):** Compare a fixed set of neighboring Sector 6 frames with the representative-frame query, controlling frame count, temporal sampling, visual token budget, and context length. The model's video support is not proof of adequate detection of subtle capsule-endoscopy findings.

**Comparison against MedVTok (Section 9):** MedVTok is a medical visual tokenizer with potentially useful domain-specific representations; EmbeddingGemma 2 is a general-purpose **joint text/visual embedding** model. Compare both in **RAG-V** under the same evidence protocol. For **RAG-MM**, compare EmbeddingGemma 2's native cross-modal representation with appropriately aligned/projected MedVTok + text embeddings, without assuming raw vector-space compatibility.

### Controlled ablations and evaluation

- Keep final split assignments, evidence IDs/provenance, question sets, index-corpus definitions, top-K, downstream generator, prompting and answer-validation rules fixed wherever possible.
- Compare **Recall@K, Precision@K, MRR, nDCG@K, evidence/segment recall, localization and temporal grounding, answer accuracy, abstention, retrieval latency, embedding throughput, index storage, and GPU memory**.
- Test embedding dimensions **768 / 512 / 256** first; include 128 only as an explicit lower-quality compression experiment. Re-normalize embeddings after truncation and ensure query and index vectors have identical dimensions.
- Evaluate the image token budget on small or low-contrast lesions; higher token budgets may preserve detail at additional computation cost.
- Report retrieval results by pathology/finding category, visual quality and question type; assess whether gains in retrieval actually improve **grounded** VQA responses.
- Maintain strict data isolation across physical frames, derivatives, temporal windows and, where feasible, source videos; never tune parameters or select configurations using the test set.
- Distinguish general multimodal benchmark results from capsule-endoscopy evidence. The model has **not been validated on Kvasir-Capsule** in the cited work.

### Practical checks before adoption

- [ ] Confirm download, applicable license/terms, and successful inference of the published checkpoint on available hardware.
- [ ] Use **bfloat16 or float32**; the official model card warns that **float16 can produce NaNs or degraded embeddings**.
- [ ] Reproduce a tiny text↔image retrieval example using the model's query/document instruction format.
- [ ] Create a small, split-safe Kvasir-Capsule index and compare with the existing general embedding baseline.
- [ ] Compare **MedVTok vs. EmbeddingGemma 2** on visual retrieval and test native image+question joint retrieval.
- [ ] Measure the impact of embedding truncation and vision token budget on subtle findings.
- [ ] Decide on further experiments using validated retrieval and grounding results, not vendor benchmark rankings alone.

**Decision:** Retain EmbeddingGemma 2 as an **open-weight multimodal retrieval candidate to investigate later**. No claims of implementation, clinically validated performance, or improvement on the MMVQA benchmark.
