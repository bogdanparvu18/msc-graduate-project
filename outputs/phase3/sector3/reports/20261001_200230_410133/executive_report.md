# Phase 2 — Sector 3: Clinical Taxonomy and Labelled-Frame Manifest

Run: `20261001_200230_410133`.

This run audits clinical-label mappings, preserves source annotations and alignment provenance, validates decoded-frame bounds, and flags bounding boxes suitable for spatial grounding. It produces frame inventories, class distributions, and a binary video/class matrix for downstream stratification.

The accepted manifest contains **47,239 annotation rows**, **47,229 distinct video frames**, and **43 labelled videos**. **4,127 annotation rows** have spatially usable bounding boxes.

Published dataset counts are diagnostic references. Frames without accepted evidence are not automatically clinically negative or unlabelled in the original dataset. A zero in the class matrix means no accepted annotation for that class. No split assignment or model inference is performed here.

The separate configuration envelope preserves both upstream configurations and lineage. Parquet manifests and readable CSV/JSON reports belong to one checksummed run. GitHub publication uses `outputs/phase3/sector3` as explicitly configured.
