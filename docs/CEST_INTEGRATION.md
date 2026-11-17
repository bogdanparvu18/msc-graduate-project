# CEST 2005 on the PR26 declarative engine

Base: `07d6a44ae03ff3573230ff574ae16286b263bc99` (PR26 and PR27 merged). CEST also reuses the shared manifest validator from PR27.

## Use

Place the authorized original `CEST.pdf` in the existing Drive terminology input
folder, or configure `cest_document_path`. Automatic discovery uses only
`["CEST.pdf"]`. No local transcription needs to be authored by the operator.
Run the notebook from section 1. After merge, the default GitHub refs are `main`.
Before merge, set **both** RAG and CEST GitHub config refs to
`feat/cest-declarative-integration` before starting the run.

The shared GitHub manifest loader retrieves a versioned transcription, verifies
its bytes/content against its manifest, and caches it by hash. The source PDF
is verified separately. Their combined fingerprint identifies the imported
source snapshot and is pinned by every CEST mapping decision. An optional local
`cest_package_path` is accepted only when it matches the verified package bytes.
No local mapping-review override is re-enabled.

## Source transcription

`cest_2005_package.json` covers the table entries from Tables 1–11: 128 local
term records and 395 attribute-value records, plus original context, table/page
locators, scoped notes and unbound values. These are counts of **this project's
representation**, not official CEST concept counts. Child qualifiers and report
fields are retained as distinct source roles. The source notice is preserved.

The p.956 right-to-left table continuation and the two p.957 column continuations
are explicitly located. Blank attribute headings are not invented: Diverticulum
values and the Abnormal imaging instruction remain unbound source values.
Foreign Body / Type has an explicit `unspecified_in_source` value. Source
categorical sizes are not measurements. The source PDF remains in Drive and is
not added to this repository.

Source-image review was performed with AI assistance. `human_reviewed=false`
and `clinical_validation_performed=false` are explicit. Byte integrity and
transcription checks do not establish the truth of a dataset annotation.

## Mapping

Eight CEST-specific rules extend the existing ordered `RAG_DECISION_PIPELINE`
after common identity/evidence/rejection guards and before lexical fallback.
No new terminology-specific `if/elif` decision chain is introduced. Source
facts and final source-constraint validation are separate helper functions.

There are 16 source-pinned project decisions: 7 exact, 8 contextual related and
1 compositional. Finding and diagnosis roles remain distinct. Only approved,
unconstrained exact maps supply interchangeable class terms. Hematin remains
an explicit Blood Type qualifier. Fresh is **not** silently equated to CEST Red;
fresh blood has only a contextual Blood association. The two native-only
classes retain their existing policies.

MST/SNOMED profiles and non-CEST target decisions are unchanged. The RAG package
hash changes because CEST evidence was added, so the existing ulcer regression
fixture's policy-reference hash is refreshed without changing its expectations.
No S8 annotation, task, geometry, RF2 importer, QA generation policy or coverage
verdict is relaxed. Non-enumerated new CEST placeholder kinds cannot supply
attribute-comparison evidence.

## Tests and limits

`python -m pytest tests/test_cest_integration.py -q`

The tests execute the notebook's functions in isolation, including candidate
preparation, declarative review, final application, source-synonym checks,
manifest/cache checks and all 891 coverage verdict/reason triples. Negative
cases cover altered source bytes, roles, fingerprints and qualifiers. Synthetic
PDF-byte fixtures exercise integrity mechanics without redistributing the PDF.
Set `CEST_TEST_PDF` to the authorized original for the real-file check. CI skips
that one check when the PDF is not supplied; it tests Parquet using pyarrow.
The complete Colab S8/RF2 run is not reproduced by these focused tests.

The authoring scripts reproduce this one-time patch from the exact base above;
they fail on another notebook baseline or an already-integrated policy. They
are not part of runtime ingestion and should not be blindly rerun after merge.

## Source notice

Copyright and Terms of Use
Copyright © 2005 Given Imaging Ltd. (“Given Imaging”). All rights reserved.
This Capsule Endoscopy Structured Terminology (CEST) is a copyrighted work, protected by United States and worldwide copyright laws and treaties. Users of the CEST are hereby granted a free license to use, copy or distribute the CEST in connection with supporting a structured terminology for capsule endoscopy, subject to the following conditions:
1. This copyright and terms of use notice must be displayed on each and every copy of the CEST, on all manuals and other materials used in connection with the CEST, including without limitation electronic media (i. e., disks, CD-ROMs, etc.), and in all text files loaded onto electronic media or the Internet. Such copyright and terms of use notice shall not be removed, obscured, modified or otherwise hidden. All references to the CEST, in whole or in part, shall properly cite the CEST as the source for such reference. Proper citation shall read “Capsule Endoscopy Structured Terminology (CEST), Endoscopy 2005; 37: 951-959 (February 16, 2002 © Given Imaging Ltd.)”.
2. The CEST shall not be used, in whole or in part, to harm or degrade the reputation of Given Imaging or any participant in the Given Capsule Standard Working Group (the “GCSWG”).
3. Neither Given Imaging nor any participant in the GCSWG shall be liable for any errors or omissions contained in the CEST, or for any use you or any third party may make of the CEST. Given Imaging reserves the right to make changes to the CEST without notice. The information contained in the CEST is provided on an “as is” basis without any warranties, express or implied, including but not limited to the results or effects obtained through use of information, or that it is fit for any use intended or can be used without infringing the copyright or other proprietary rights of any third party. All express and implied warranties, including those relating to title, non-infringement, merchantability or fitness for a particular purpose, are hereby expressly disclaimed and excluded with respect to any information provided hereunder.
