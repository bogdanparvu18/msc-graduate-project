# Executive report: Phase 3, Sector 1 — Clinical terminology mapping and evidence contract

**Source:** [`phase3_01_terminology_mapping.ipynb`](../../../notebooks/phase3/phase3_01_terminology_mapping.ipynb)  
**Review date:** 18 November 2026  
**Basis:** The Colab notebook and the published Sector 1 bundle selected by [`latest.json`](latest.json). The reviewed run is `20261118T013916_145151Z` at data commit [`5a2a57da0b42dbcb7a33a848d4f1c1aafadab58c`](https://github.com/bogdanparvu18/msc-graduate-project/commit/5a2a57da0b42dbcb7a33a848d4f1c1aafadab58c). The notebook retains partial cell output. The run summary, crosswalk, coverage matrix and publication manifests supply the final counts below. Drive verification is reported as recorded by the publisher.

## Executive summary

Sector 1 links the **14 classes** annotated in Kvasir-Capsule to medical terminology and defines the conditions under which Sector 2 can use those mappings to construct a question and answer. Its inputs are the Phase 2 Sector 8 recipes and annotation evidence. The inventory covers **47,239 annotations on 47,229 distinct frames**. Each class keeps its original label and annotation provenance alongside its terminology mappings.

The run imported **MST 3.0, CEST 2005, the Canadian Edition of SNOMED CT and the 2019 capsule vascular nomenclature supplement**. The crosswalk contains **55 rows: 48 approved, one rejected and six unmapped**. Approved rows distinguish exact, related and compositional relations. The vocabulary export records 91 approved synonym rows with provenance, representing 30 distinct surface forms. Approval records the outcome of the project's evidence-backed mapping policy. The run records `human_reviewed=false`. [Mapping review summary](metadata/20261118T013916_145151Z/reports/rag_review_summary.json)

**The Sector 2 handoff is published.** It includes the crosswalk, the expected coverage matrix, annotation provenance and an executable validator. The matrix specifies 891 combinations of task, input variant and evidence state. For a particular question, Sector 2 must also check the selected recipe and its supporting annotations. Sector 1 generated **zero QA pairs**, and observed QA coverage remains unevaluated. [Run summary](metadata/20261118T013916_145151Z/reports/summary.json)

## Workflow

| Stage | What the notebook does |
| --- | --- |
| Phase 2 source selection | Resolves the upstream repository's default branch and follows the S8 storage index. The saved snapshot records commit `18cd001f6ba7951e32b1bb82197ece9920e4e8ec`, the catalog SHA-256 and the identities of 19 consumed artifacts. |
| Recipe and annotation inventory | Checks the joins between recipes, ordered visual inputs, source frames and subjects across the seven S8 tables. Normalizes label case and whitespace, retains annotation IDs and bounding-box provenance, and inventories explicitly verified observation attributes. Pixel evidence comes from the completed S8 snapshot. |
| Expected coverage | Evaluates the task policy for each combination of nine tasks, eleven S8 variants and nine evidence states. Each row retains its expected status, rule and reason for later inspection. |
| Terminology import | Parses the MST workbook, the CEST transcription linked by hash to its source document, the SNOMED CT RF2 Snapshot and the vascular supplement. The run records matching Drive and VM SHA-256 values for the MST workbook, CEST document, CEST package and RF2 source. The GitHub manifests separately identify the RAG evidence and GI supplement. |
| Mapping and vocabulary | Retrieves source-linked candidates and applies the versioned RAG policy. Records the target identity, relation, supporting evidence and decision. Accepted source descriptions of approved exact concepts can enter the interchangeable vocabulary. Related concepts and required attribute values retain their own scope. |
| Contract enrichment | Attaches the class mappings to clinical roles in the task and evidence contract. Preserves the original coverage decisions and records which capabilities have supporting observations. A concrete request supplies the recipe, task, evidence state and annotation operands to the binding validator. |
| Export and publication | Writes the tables, audits and Sector 2 validator. The published run groups files into `configs/`, `reports/` and `results/`. The publication manifest records GitHub paths, sizes and SHA-256 values. A separate manifest locates the Parquet files verified in Drive. |

For example, the source class `ampulla of vater` has an approved **exact** SNOMED CT mapping to *Structure of ampulla of Vater* and an approved **related** MST mapping to *Major papilla*. The relation controls whether a target term can supply interchangeable wording. The annotation ID and its presence in the displayed recipe establish which frame can support a question. [Class crosswalk](metadata/20261118T013916_145151Z/results/class_crosswalk.json)

## Results visible in the saved notebook

The saved Section 2 output shows the upstream recipe inventory, and Section 4 shows the expected coverage matrix. The run-specific reports supply the final terminology and export figures for 18 November. [S8 metadata validation](metadata/20261118T013916_145151Z/reports/phase2_metadata_validation.json) · [Phase 2 inventory](metadata/20261118T013916_145151Z/reports/phase2_inventory_summary.json) · [Sector 1 summary](metadata/20261118T013916_145151Z/reports/summary.json)

| Measure | Published result |
| --- | ---: |
| Phase 2 annotations / distinct annotated frames / classes | 47,239 / 47,229 / 14 |
| S8 recipes / ordered input slots / distinct referenced visual artifacts | 216,808 / 309,381 / 184,328 |
| Imported terminology sources | 4 |
| Terminology term rows / attribute-schema rows | 662 / 528 |
| Crosswalk rows / rows with an accepted or rejected review | 55 / 49 |
| Approved exact / related / compositional mappings | 25 / 20 / 3 |
| Approved synonym rows / distinct surface forms | 91 / 30 |
| Expected combinations — applicable / not applicable / review required | 158 / 455 / 278 |
| Terminology schema bindings / evidence capability rows | 4,675 / 158 |
| Semantic compositions declared / executed | 203 / 0 |
| Verified observation-attribute records / new clinical observations / QA pairs | 0 / 0 / 0 |

The exported terminology table contains 662 term rows. A Phase 2 class can have several target concepts or relations in one terminology, so the crosswalk counts each mapping decision separately. [Terminology terms](metadata/20261118T013916_145151Z/results/terminology_terms.json)

| Terminology | Exported term rows | Approved maps | Rejected maps | Unmapped rows | Crosswalk rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| MST 3.0 | 368 | 13 | 1 | 2 | 16 |
| CEST 2005 | 128 | 16 | 0 | 2 | 18 |
| SNOMED CT | 162 | 18 | 0 | 2 | 20 |
| SBCE vascular nomenclature, 2019 | 4 | 1 | 0 | 0 | 1 |
| **Total** | **662** | **48** | **1** | **6** | **55** |

The six unmapped rows belong to `normal clean mucosa` and `reduced mucosal view`, each across MST, CEST and SNOMED CT. For `angiectasia`, the review rejected the MST *related* candidate and approved the *exact* mapping. In the Canadian SNOMED CT release dated 31 August 2026, the importer retained 93 description issues affecting 85 concepts for audit. The exported SNOMED subset contains 162 concepts and 225 relationships. [Crosswalk](metadata/20261118T013916_145151Z/results/class_crosswalk.json) · [SNOMED description audit](metadata/20261118T013916_145151Z/reports/snomed_description_audit.json)

The 891 expected decisions reconcile as **9 tasks × 11 variants × 9 evidence states**. The 4,675 bindings describe where a mapping can be referenced in the contract. Observation-level support must still be checked for a specific request. For example, the terminology supplies *Size* as an attribute for `polyp`, but the published capability table records **zero** verified values for a size comparison. Sector 2 can use that attribute name in its vocabulary. Giving a numerical answer about a particular polyp requires a measured value tied to that observation. [Coverage contract](metadata/20261118T013916_145151Z/results/coverage_contract.json) · [Evidence capabilities](metadata/20261118T013916_145151Z/results/phase2_evidence_capabilities.json)

## Outputs and scope

The run is published under `metadata/20261118T013916_145151Z/`. Start with the sector-level [`latest.json`](latest.json) to obtain the selected run ID, data commit, results directory and manifest digests. Resolve the paths against that commit so the contract, code and data come from one publication.

| Artifact | Purpose |
| --- | --- |
| [`publication_manifest.json`](metadata/20261118T013916_145151Z/publication_manifest.json) | Maps logical filenames to GitHub paths, sizes and SHA-256 values. Lists required Sector 2 files and records the reason for each omitted entry. |
| [`configs/drive_artifacts_manifest.json`](metadata/20261118T013916_145151Z/configs/drive_artifacts_manifest.json) | Locates the 39 Parquet files beneath the configured Drive storage root. Each entry carries the byte count, SHA-256 and recorded VM/Drive comparison. |
| [`configs/source_manifest.json`](metadata/20261118T013916_145151Z/configs/source_manifest.json) and [`configs/attribute_registry.json`](metadata/20261118T013916_145151Z/configs/attribute_registry.json) | Identify the imported terminology sources and define the supported attributes, units and source bindings. |
| [`reports/summary.json`](metadata/20261118T013916_145151Z/reports/summary.json) and [`reports/contract_binding_audit.json`](metadata/20261118T013916_145151Z/reports/contract_binding_audit.json) | Report run status, counts and the integrity of the original task and evidence policies after terminology enrichment. |
| [`results/class_crosswalk.json`](metadata/20261118T013916_145151Z/results/class_crosswalk.json) and [`results/approved_class_synonyms.json`](metadata/20261118T013916_145151Z/results/approved_class_synonyms.json) | Supply class-to-concept decisions and the vocabulary eligible for interchangeable wording. Source-attested alternatives are available in a separate table. |
| [`results/phase2_classes.json`](metadata/20261118T013916_145151Z/results/phase2_classes.json), [`phase2_annotation_provenance.json`](metadata/20261118T013916_145151Z/results/phase2_annotation_provenance.json) and [`phase2_source_snapshot.json`](metadata/20261118T013916_145151Z/results/phase2_source_snapshot.json) | Connect class IDs to the original S8 annotations and identify the upstream revision and catalog used to construct the inventory. |
| [`results/coverage_contract.json`](metadata/20261118T013916_145151Z/results/coverage_contract.json) and [`results/contract_terminology_bindings.json`](metadata/20261118T013916_145151Z/results/contract_terminology_bindings.json) | Supply the expected decision matrix and the terminology bindings addressable by the runtime. |
| [`results/semantic_compositions.json`](metadata/20261118T013916_145151Z/results/semantic_compositions.json) and [`results/phase2_evidence_capabilities.json`](metadata/20261118T013916_145151Z/results/phase2_evidence_capabilities.json) | Describe the declared operations and the available evidence for each class and capability. |
| [`results/sector2_binding_contract.json`](metadata/20261118T013916_145151Z/results/sector2_binding_contract.json) and [`results/sector2_binding_validator.py`](metadata/20261118T013916_145151Z/results/sector2_binding_validator.py) | Describe the required corpus interface and provide the executable loader and eligibility checks for Sector 2. |
| [`results/files_manifest.json`](metadata/20261118T013916_145151Z/results/files_manifest.json) | Records the original run artifacts with their byte counts and SHA-256 values. Its own trusted digest is in the publication pointer. |

The run manifest lists **119 artifacts**. GitHub publishes 79 of those payloads. Another 39 are Parquet files retained in verified Drive storage. The remaining entry is the original MST workbook, which the publisher marked `unsupported_extension`. Adding the run and Drive manifests to the 79 published payloads gives **81 files listed by the publication manifest**: 25 under `configs/`, 20 under `reports/` and 36 under `results/`. The publication manifest itself and `latest.json` are outside that count.

**Sector 2 import.** All nine required files are available in `results/`. Verify `files_manifest.json` against the SHA-256 in `latest.json`, then verify the validator module against its manifest entry **before importing it**. Call `load_sector1_binding_contract(directory, corpus, expected_manifest_sha256=...)` with the verified S8 corpus. The loader checks its six required data files. For each proposed QA request, call `require_eligible(...)` with the recipe, binding ID, task, evidence state and annotation IDs, plus reference operands when the task requires them. If using a Parquet copy from Drive, verify the Drive manifest against its digest in `latest.json`, then compare the file's size and SHA-256 with its entry. [Exported interface and usage](metadata/20261118T013916_145151Z/results/sector2_binding_contract.json)

The exported validator passed its synthetic supported-annotation and unsupported-annotation cases. Those tests exercise the generated module. Clinical review, image-level verification of new answers and observed QA coverage remain downstream work. The published run records `qa_generated=0`, `ready_for_generation=false` and `qa_generation_authorized=false`.
