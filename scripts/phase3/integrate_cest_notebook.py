"""Apply the reviewed CEST integration to the PR26 notebook, without a broad reformat.

Source text is replaced only at uniquely matched anchors. Clinical data live in
cest_2005_package.json and the versioned RAG package, never in this patch logic.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = 'notebooks/phase3/phase3_01_terminology_mapping.ipynb'
BASE_BLOB = '0fdba69fc91f77be3720a55fe9b0e5392f936534'

ACQUISITION = '''CEST_DOCUMENT_DRIVE_PATH = acquire_ontology_input(
    "CEST_DOCUMENT", TERMINOLOGY_CONFIG["cest_document_path"],
    TERMINOLOGY_CONFIG["cest_document_patterns"], "CEST.pdf")
CEST_SOURCE_PROVENANCE = None
SOURCE_MANIFESTS["CEST"] = {"status": "source_required", "document_status": "missing",
    "transcription_status": "not_loaded", "reason": "The original CEST.pdf and its verified transcription are required",
    "source_url": "https://doi.org/10.1055/s-2005-870329"}
if CEST_DOCUMENT_DRIVE_PATH is not None:
    CEST_DOCUMENT_PATH = stage_terminology_input(CEST_DOCUMENT_DRIVE_PATH, "CEST_DOCUMENT")
    SOURCE_MANIFESTS["CEST"]["document_status"] = "stored_and_verified"
    CEST_PACKAGE, CEST_SOURCE_PROVENANCE = load_github_cest_package(
        repository=CEST_GITHUB_CONFIG["repository"], ref=CEST_GITHUB_CONFIG["ref"],
        manifest_path=CEST_GITHUB_CONFIG["manifest_path"], cache_root=WORK / "cest_github_cache",
        fetch=STORE._download, attempts=CEST_GITHUB_CONFIG["download_attempts"])
    CEST_PACKAGE_REFERENCE = Path(CEST_SOURCE_PROVENANCE["local_path"])
    # A local package may be supplied, but never silently override the verified transcription.
    if TERMINOLOGY_CONFIG["cest_package_path"]:
        CEST_DRIVE_PATH = Path(TERMINOLOGY_CONFIG["cest_package_path"]).expanduser().resolve()
        verify_file(CEST_DRIVE_PATH, CEST_SOURCE_PROVENANCE["artifact_bytes"], CEST_SOURCE_PROVENANCE["source_sha256"])
    else:
        CEST_DRIVE_PATH = TERM_REGISTRY_ROOT / "CEST_PACKAGE" / CEST_SOURCE_PROVENANCE["source_sha256"] / "cest_2005_package.json"
        CEST_DRIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not CEST_DRIVE_PATH.exists():
            pending = _rag_github_temporary(CEST_DRIVE_PATH.parent)
            try:
                shutil.copy2(CEST_PACKAGE_REFERENCE, pending)
                verify_file(pending, CEST_SOURCE_PROVENANCE["artifact_bytes"], CEST_SOURCE_PROVENANCE["source_sha256"])
                os.replace(pending, CEST_DRIVE_PATH)
            finally:
                pending.unlink(missing_ok=True)
        verify_file(CEST_DRIVE_PATH, CEST_SOURCE_PROVENANCE["artifact_bytes"], CEST_SOURCE_PROVENANCE["source_sha256"])
    CEST_DRIVE_PATH = acquire_ontology_input("CEST_PACKAGE", CEST_DRIVE_PATH, [], "cest_2005_package.json")
    CEST_PATH = stage_terminology_input(CEST_DRIVE_PATH, "CEST_PACKAGE")
    SOURCE_BUNDLES["CEST"] = read_cest_package(CEST_PATH, source_document_path=CEST_DOCUMENT_PATH)
    if SOURCE_BUNDLES["CEST"]["manifest"]["source_fingerprint"] != CEST_SOURCE_PROVENANCE["source_fingerprint"]:
        raise ValueError("CEST imported source pair differs from its verified GitHub manifest")
    SOURCE_STATUS["CEST"] = "imported"
    SOURCE_MANIFESTS["CEST"] = SOURCE_BUNDLES["CEST"]["manifest"]
    SOURCE_MANIFESTS["CEST"].update(document_status="verified", transcription_status="verified_AI_review_not_human",
        github_provenance=CEST_SOURCE_PROVENANCE,
        source_artifacts={key: SOURCE_STAGING[key] for key in ("CEST_PACKAGE", "CEST_DOCUMENT")})
    remember_ontology_ids("CEST_PACKAGE", {"version": SOURCE_MANIFESTS["CEST"]["version"],
        "source_fingerprint": SOURCE_MANIFESTS["CEST"]["source_fingerprint"],
        "document_source_sha256": SOURCE_STAGING["CEST_DOCUMENT"]["source_sha256"],
        "identifier_scope": "project_local_source_locators_not_official_CEST_codes",
        "term_ids": SOURCE_BUNDLES["CEST"]["terms"]["term_id"].tolist()})
    write_json(RUN_DIR / "cest_source_provenance.json", CEST_SOURCE_PROVENANCE)

'''


def apply(root=ROOT):
    root=Path(root); path=root/NOTEBOOK; data=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if blob!=BASE_BLOB:
        raise ValueError('Notebook baseline changed; review the patch against current main instead of force-applying it')
    notebook=json.loads(data); original=copy.deepcopy(notebook)
    def get(i): return ''.join(notebook['cells'][i]['source'])
    def put(i,s):
        c=notebook['cells'][i];c['source']=s.splitlines(keepends=True)
        if c['cell_type']=='code':
            c['outputs']=[];c['execution_count']=None
            c['metadata'].pop('outputId',None);c['metadata'].pop('colab',None)
    def replace(i,a,b):
        s=get(i)
        if s.count(a)!=1:raise ValueError(f'Expected one patch anchor in cell {i}: {a[:90]}')
        put(i,s.replace(a,b,1))
    replace(3,'    "cest_package_path": None,  # Drive JSON; its source_file points to the original document.',
        '    "cest_package_path": None,  # Optional local copy; must match the verified GitHub transcription.\n'
        '    "cest_document_path": None,  # Original authorized CEST PDF in Drive.\n'
        '    "cest_document_patterns": ["CEST.pdf"],')
    replace(3,'WORK = Path(CONFIG["work_root"]).expanduser().resolve()',
        'CEST_GITHUB_CONFIG = {\n    "repository": CONFIG["repository"],\n'
        '    "ref": RAG_GITHUB_CONFIG["ref"],\n'
        '    "manifest_path": "outputs/phase3/sector1/terminology/cest_2005_manifest.json",\n'
        '    "download_attempts": 3,\n}\n\nWORK = Path(CONFIG["work_root"]).expanduser().resolve()')
    runtime=(root/'scripts/phase3/cest_runtime.py').read_text()
    replace(11,'def read_cest_package(path, source_document_path=None):',runtime+'\n\ndef read_cest_package(path, source_document_path=None):')
    replace(11,'    payload = json.loads(package_bytes.decode("utf-8-sig"))',
        '    payload = validate_cest_transcription(_rag_github_json(package_bytes))')
    replace(11,'    source_hash = hashlib.sha256(original.read_bytes()).hexdigest()',
        '    original_bytes = original.read_bytes()\n'
        '    if not original_bytes.startswith(b"%PDF-") or len(original_bytes) != metadata["source_bytes"]:\n'
        '        raise ValueError("CEST original document format or byte count differs from the transcription")\n'
        '    source_hash = hashlib.sha256(original_bytes).hexdigest()')
    replace(11,'            "applicability": _mst_table([], MST_APPLICABILITY_COLUMNS),\n            "manifest": {**metadata, "terminology": "CEST",',
        '            "applicability": _mst_table([], MST_APPLICABILITY_COLUMNS),\n'
        '            "source_contexts": pd.DataFrame(payload["source_contexts"]).assign(terminology="CEST"),\n'
        '            "source_notes": pd.DataFrame(payload.get("source_notes", [])),\n'
        '            "unbound_source_values": pd.DataFrame([\n'
        '                {**{k:v for k,v in row.items() if k != "values"}, "values_json": canonical_json(row["values"])}\n'
        '                for row in payload.get("unbound_source_values", [])]),\n'
        '            "manifest": {**metadata, "terminology": "CEST",')
    replace(11,'"parser_version": "cest-reviewed-transcription-v1"','"parser_version": "cest-source-context-transcription-v2"')
    replace(11,'        "terms": [], "synonyms": [], "attributes": []}',
        '        "terms": [], "synonyms": [], "attributes": [], "source_contexts": [],\n'
        '        "source_notes": [], "unbound_source_values": []}')
    replace(11,'"copyright_notice": "", "transcription_reviewed_by": "", "transcription_reviewed_at": ""}',
        '"copyright_notice": "", "transcription_reviewed_by": "", "transcription_reviewed_at": "",\n'
        '        "source_bytes": None, "page_count": 9, "human_reviewed": False}')
    s=get(15);start=s.index('CEST_DRIVE_PATH = acquire_ontology_input(');end=s.index('SNOMED_DRIVE_PATH = acquire_ontology_input(',start)
    put(15,s[:start]+ACQUISITION+s[end:])
    replace(15,'    snomed_candidates = pd.DataFrame()\n    snomed_retrieval_audit',
        '    cest_candidates, cest_proposal_audit = cest_pinned_candidates(classes, bundles, evidence_bundle)\n'
        '    candidates.extend(cest_candidates)\n\n    snomed_candidates = pd.DataFrame()\n    snomed_retrieval_audit')
    replace(15,'        "class_rows": class_rows, "classes_by_id": classes_by_id, "classes_by_label": classes_by_label,',
        '        "cest_proposal_audit": cest_proposal_audit,\n'
        '        "cest_pinned_decisions": copy.deepcopy([d for d in (evidence_bundle or {}).get("target_decisions", []) if d.get("terminology") == "CEST"]),\n'
        '        "class_rows": class_rows, "classes_by_id": classes_by_id, "classes_by_label": classes_by_label,')
    replace(15,'        if decision == "approved":\n            if terminology == "SNOMED_CT"',
        '        if decision == "approved":\n'
        '            source_validator = CEST_REVIEW_VALIDATORS.get(terminology)\n'
        '            if source_validator is not None:\n'
        '                source_validator(review, context, term)\n'
        '            if terminology == "SNOMED_CT"')
    replace(15,'            "mst_proposal_audit": context["mst_proposal_audit"]}',
        '            "mst_proposal_audit": context["mst_proposal_audit"],\n'
        '            "cest_proposal_audit": context["cest_proposal_audit"]}')
    # Extend rule data, preserving every PR26 rule and its relative ordering.
    replace(18,'RAG_DECISION_FACT_FIELDS = (','RAG_DECISION_FACT_FIELDS = (\n    "cest_source_context_valid", "cest_constraints_attested", "cest_match_supported",')
    replace(18,'def build_rag_mapping_reviews(candidate_result, evidence_bundle, snomed_snapshot=None):',
        '# CEST rules execute after the common gates and before lexical fallback.\n'
        '_cest_insert_at = next(i for i, rule in enumerate(RAG_DECISION_PIPELINE) if rule["rule_id"] == "unsupported_terminology")\n'
        'RAG_DECISION_PIPELINE[_cest_insert_at:_cest_insert_at] = copy.deepcopy(CEST_DECISION_RULES)\n\n\n'
        'def build_rag_mapping_reviews(candidate_result, evidence_bundle, snomed_snapshot=None):')
    replace(18,'        result = rag_apply_decision_pipeline(context, RAG_DECISION_PIPELINE, RAG_DECISION_ACTIONS)',
        '        context.update(cest_decision_facts(\n'
        '            row, decision, term, candidate_result.get("_prepared_mapping_context", {}).get("bundles", {}).get("CEST")))\n'
        '        result = rag_apply_decision_pipeline(context, RAG_DECISION_PIPELINE, RAG_DECISION_ACTIONS)')
    # Undefined source values cannot become comparability evidence.
    replace(20,'d.get("value_kind") == "classification_reference"',
        'd.get("value_kind") in {"classification_reference", "free_text_placeholder", "source_table_reference", "unspecified_in_source"}')
    replace(23,'    "mst_proposal_audit": MAPPING_RESULT["mst_proposal_audit"],',
        '    "mst_proposal_audit": MAPPING_RESULT["mst_proposal_audit"],\n'
        '    "cest_proposal_audit": MAPPING_RESULT["cest_proposal_audit"],')
    replace(23,'for name, table in TABLES_TO_SAVE.items():\n    terminology_write_table(TERM_OUTPUT, name, table)',
        'if "CEST" in SOURCE_BUNDLES:\n'
        '    for key in ("source_contexts", "source_notes", "unbound_source_values"):\n'
        '        TABLES_TO_SAVE["cest_" + key] = SOURCE_BUNDLES["CEST"][key].assign(\n'
        '            copyright_notice=SOURCE_MANIFESTS["CEST"]["copyright_notice"])\n'
        'for name, table in TABLES_TO_SAVE.items():\n'
        '    if "CEST" in SOURCE_BUNDLES and "terminology" in table and table["terminology"].eq("CEST").any():\n'
        '        table = table.copy()\n'
        '        table["copyright_notice"] = table["terminology"].map(\n'
        '            lambda value: SOURCE_MANIFESTS["CEST"]["copyright_notice"] if value == "CEST" else None)\n'
        '    terminology_write_table(TERM_OUTPUT, name, table)')
    replace(23,'if "CEST" in SOURCE_BUNDLES:\n    (TERM_OUTPUT / "CEST_COPYRIGHT.txt").write_text',
        'if "CEST" in SOURCE_BUNDLES:\n'
        '    verify_file(CEST_PATH, CEST_SOURCE_PROVENANCE["artifact_bytes"], CEST_SOURCE_PROVENANCE["source_sha256"])\n'
        '    verify_file(CEST_DOCUMENT_PATH, SOURCE_MANIFESTS["CEST"]["source_bytes"], SOURCE_MANIFESTS["CEST"]["original_document_sha256"])\n'
        '    shutil.copy2(CEST_PATH, TERM_OUTPUT / "sources" / "cest_2005_package.json")\n'
        '    shutil.copy2(CEST_SOURCE_PROVENANCE["manifest_local_path"], TERM_OUTPUT / "cest_2005_manifest.json")\n'
        '    write_json(TERM_OUTPUT / "cest_source_provenance.json", CEST_SOURCE_PROVENANCE)\n'
        '    write_json(TERM_OUTPUT / "cest_decision_rules.json", {"rules": CEST_DECISION_RULES,\n'
        '        "rules_sha256": canonical_sha256(CEST_DECISION_RULES), "copyright_notice": SOURCE_MANIFESTS["CEST"]["copyright_notice"]})\n'
        '    (TERM_OUTPUT / "CEST_COPYRIGHT.txt").write_text')
    replace(23,'"gi_github": GI_GITHUB_CONFIG})','"gi_github": GI_GITHUB_CONFIG, "cest_github": CEST_GITHUB_CONFIG})')
    # Correct only CEST documentation, plus the already-disabled override sentence nearby.
    s=get(2).replace('`cest_package.json` plus its referenced document','`CEST.pdf` plus a manifest-verified CEST transcription from GitHub')
    s=s.replace('`cest_source.pdf`','`CEST.pdf`').replace('An explicit review override is optional; the default is the RAG evidence package fetched from GitHub.',
        'Local mapping-review overrides remain disabled; the verified GitHub RAG package supplies mapping decisions.')
    put(2,s)
    put(10,get(10)+'\nCEST 2005 is transcribed from the supplied PDF, with Tables 1–11 and source-role locators retained. The PDF is not parsed during a run. Transcription review is AI-assisted, not human or clinical validation. Local IDs are not official CEST codes. Blank attribute cells and source-defined categorical sizes remain explicit.\n')
    put(14,get(14)+'\nCEST discovery uses only `CEST.pdf`, or `cest_document_path`. The matching JSON is fetched through `CEST_GITHUB_CONFIG` and cached by hash. An optional local `cest_package_path` must match those verified bytes. A missing PDF remains `source_required`; a mismatched PDF fails closed. To test an unmerged PR, set both RAG and CEST GitHub refs to its branch before running from section 1.\n')
    put(17,get(17)+'\nCEST decisions extend the existing ordered declarative pipeline. Explicit source-pinned targets preserve finding/diagnosis roles and compositional qualifiers. Retrieval, source import and mapping approval remain separate. `blood fresh` links only contextually to Blood: the CEST value Red is not silently equated with Fresh.\n')
    # Serialize only source/cleared-output changes, preserving all other notebook objects.
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    changed=[i for i,(a,b) in enumerate(zip(original['cells'],notebook['cells'])) if a!=b]
    print('Changed cells:',changed)
    assert all(original['cells'][i]==notebook['cells'][i] for i in [5,7,9,13,16,21])
    return changed

if __name__=='__main__': apply()
