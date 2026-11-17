"""CEST helpers embedded in the notebook by integrate_cest_notebook.py.

Uses the notebook's existing JSON, table, storage and candidate functions.
No PDF parsing, network I/O at import, mapping approval or clinical inference.
"""

CEST_VALUE_KINDS = frozenset({
    'enumerated', 'free_text_placeholder', 'source_table_reference', 'unspecified_in_source',
})


def validate_cest_transcription(payload):
    if not isinstance(payload, dict) or payload.get('schema_version') != 'mmvqa-cest-transcription-v1':
        raise ValueError('Unsupported CEST transcription schema')
    metadata = payload.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError('CEST metadata must be an object')
    for key in ('version', 'citation', 'source_url', 'source_file', 'copyright_notice',
                'transcription_reviewed_by', 'transcription_reviewed_at'):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError('CEST metadata missing: ' + key)
    _rag_github_sha(metadata.get('source_sha256'), 'CEST original document SHA-256')
    if type(metadata.get('source_bytes')) is not int or metadata['source_bytes'] <= 0:
        raise ValueError('CEST original document byte count is required')
    if type(metadata.get('human_reviewed')) is not bool:
        raise ValueError('CEST transcription requires an explicit human-review flag')
    date.fromisoformat(metadata['transcription_reviewed_at'])
    records = payload.get('terms')
    if not isinstance(records, list) or not records:
        raise ValueError('CEST transcription contains no terms')
    ids = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('CEST terms must be objects')
        for key in ('term_id', 'preferred_term', 'term_type', 'section', 'source_locator'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('CEST term missing: ' + key)
        if row['term_id'] in ids:
            raise ValueError('Duplicate CEST term ID')
        ids.add(row['term_id'])
    contexts = payload.get('source_contexts')
    if not isinstance(contexts, list) or len(contexts) != len(records):
        raise ValueError('Every CEST term requires one source context')
    seen = set()
    by_id = {r['term_id']: r for r in records}
    for context in contexts:
        if not isinstance(context, dict) or context.get('term_id') not in ids or context['term_id'] in seen:
            raise ValueError('Duplicate or unknown CEST source context')
        seen.add(context['term_id'])
        if any(context.get(k) != by_id[context['term_id']][k]
               for k in ('preferred_term','term_type','section','source_locator')):
            raise ValueError('CEST source context does not match its term')
        if type(context.get('table')) is not int or not 1 <= context['table'] <= 11:
            raise ValueError('CEST table locator is invalid')
        if type(context.get('pdf_page')) is not int or not 1 <= context['pdf_page'] <= metadata.get('page_count',0):
            raise ValueError('CEST page locator is invalid')
        if context.get('parent_term_id') is not None and context['parent_term_id'] not in ids:
            raise ValueError('CEST parent reference is invalid')
    attributes = payload.get('attributes')
    if not isinstance(attributes, list) or not isinstance(payload.get('synonyms'), list):
        raise ValueError('CEST attributes and synonyms must be lists')
    attribute_keys = set()
    for row in attributes:
        if (not isinstance(row, dict) or row.get('term_id') not in ids
                or not isinstance(row.get('attribute'),str) or not row['attribute'].strip()
                or not isinstance(row.get('source_locator'),str) or not row['source_locator'].strip()
                or row.get('value_kind') not in CEST_VALUE_KINDS or not isinstance(row.get('value'),str)):
            raise ValueError('Invalid CEST source attribute')
        if not row['value'] and row['value_kind'] != 'unspecified_in_source':
            raise ValueError('Only an explicitly unspecified CEST value may be blank')
        key = (row['term_id'],row['attribute'],row['value'],row['value_kind'])
        if key in attribute_keys:
            raise ValueError('Duplicate CEST source attribute')
        attribute_keys.add(key)
    for row in payload['synonyms']:
        if (not isinstance(row, dict) or row.get('term_id') not in ids
                or row.get('relation') != 'exact_synonym'
                or any(not isinstance(row.get(k),str) or not row[k].strip()
                       for k in ('synonym','evidence','source_locator'))):
            raise ValueError('CEST synonym must be explicitly source-attested')
    return payload


def _cest_github_manifest(data):
    manifest = _rag_github_json(data)
    if not isinstance(manifest,dict) or manifest.get('schema_version') != 'mmvqa-cest-github-manifest-v1':
        raise ValueError('Unsupported CEST manifest')
    artifact = manifest.get('artifact')
    if not isinstance(artifact,dict):
        raise ValueError('CEST manifest artifact missing')
    _rag_github_path(artifact.get('path'))
    _rag_github_sha(artifact.get('sha256'),'artifact.sha256')
    if type(artifact.get('bytes')) is not int or not 0 < artifact['bytes'] <= 8 * 1024 * 1024:
        raise ValueError('CEST artifact size is invalid')
    for key in ('package_content_sha256','original_document_sha256','source_fingerprint'):
        _rag_github_sha(manifest.get(key),key)
    expected = hashlib.sha256(b'mmvqa-cest-source-snapshot-v1\0' +
        bytes.fromhex(manifest['original_document_sha256']) + bytes.fromhex(artifact['sha256'])).hexdigest()
    if expected != manifest['source_fingerprint']:
        raise ValueError('CEST manifest fingerprint does not identify its document and package')
    return manifest


def _cest_github_package(data, manifest):
    artifact = manifest['artifact']
    if len(data) != artifact['bytes'] or hashlib.sha256(data).hexdigest() != artifact['sha256']:
        raise ValueError('CEST package bytes differ from the verified manifest')
    payload = validate_cest_transcription(_rag_github_json(data))
    if (canonical_sha256(payload) != manifest['package_content_sha256']
            or payload['metadata']['source_sha256'] != manifest['original_document_sha256']
            or payload['metadata']['source_bytes'] != manifest['original_document_bytes']):
        raise ValueError('CEST package provenance differs from the verified manifest')
    return payload


def load_github_cest_package(repository, ref, manifest_path, cache_root, fetch, attempts=3):
    return load_github_verified_artifact(
        dict(repository=repository, ref=ref, manifest_path=manifest_path, cache_root=cache_root,
             fetch=fetch, attempts=attempts, label='CEST', artifact_label='CEST transcription',
             cache_namespace='cest', provenance_schema='mmvqa-cest-github-source-v1',
             provenance_fields=('package_content_sha256','original_document_sha256','source_fingerprint')),
        manifest_validator=_cest_github_manifest, artifact_validator=_cest_github_package)


def cest_pinned_candidates(classes, bundles, evidence_bundle):
    """Build candidates and an audit, never approvals, from source-linked CEST data."""
    decisions = [d for d in (evidence_bundle or {}).get('target_decisions',[]) if d.get('terminology')=='CEST']
    classes_by_label = {terminology_key(r['normalized_label']):r for r in classes.to_dict('records')}
    source = bundles.get('CEST')
    registry = {} if source is None else {r['term_id']:r for r in source['terms'].to_dict('records')}
    candidates, audit = [], []
    for d in decisions:
        cls = classes_by_label.get(terminology_key(d.get('class_label','')))
        if cls is None:
            continue
        target = registry.get(d['target_id'])
        status = ('source_required' if source is None else 'target_missing' if target is None else
                  'source_fingerprint_mismatch' if d.get('source_sha256') != target['source_sha256'] else
                  'target_identity_mismatch' if any(d.get(a)!=target[b] for a,b in
                    (('target_term','preferred_term'),('target_type','term_type'),('source_locator','source_locator'))) else
                  'candidate_proposed')
        constraints = parse_attribute_constraints(d.get('required_attribute_constraints',[]))
        if status == 'candidate_proposed' and not cest_constraints_attested(source, target['term_id'], constraints):
            status = 'source_attribute_constraint_missing'
        audit.append(dict(class_label=cls['normalized_label'], target_id=d['target_id'], status=status,
                          expected_source_sha256=d.get('source_sha256'),
                          imported_source_sha256=None if target is None else target['source_sha256']))
        if status == 'candidate_proposed':
            candidates.append(terminology_candidate(cls,target,d['relation'],'source_located_cest_policy',constraints,d['rationale']))
    return candidates, pd.DataFrame(audit,columns=['class_label','target_id','status','expected_source_sha256','imported_source_sha256'])


def cest_constraints_attested(source, target_id, constraints):
    if source is None:
        return False
    for c in constraints:
        table = source['attributes']
        found = table.loc[table['term_id'].eq(target_id) & table['attribute'].eq(c['attribute']) & table['value'].eq(c['value'])
                          & table['value_kind'].eq('enumerated')]
        if found.empty or not found['source_sha256'].eq(source['manifest']['source_fingerprint']).all():
            return False
    return True


def cest_decision_facts(row, decision, term, source):
    valid = (source is not None and term is not None and decision is not None
             and source['manifest'].get('source_fingerprint') == row['source_sha256']
             and decision.get('source_locator') == term.get('source_locator'))
    return {'cest_source_context_valid': bool(valid),
            'cest_constraints_attested': bool(valid and cest_constraints_attested(source,row['target_id'],row['constraints'])),
            'cest_match_supported': row['match_method'] in {'exact_lexical_candidate','source_located_cest_policy'}}


def validate_cest_applied_review(review, context, term):
    """Final gate: a caller cannot bypass the verified CEST policy through reviews."""
    matches = [d for d in context.get('cest_pinned_decisions',[]) if
        terminology_key(d['class_label']) == terminology_key(review.get('class_label','')) and d['target_id']==review['target_id']]
    if len(matches)!=1:
        raise ValueError('CEST review requires one source-pinned policy decision')
    d=matches[0]
    if (review.get('decision') != d['decision'] or review.get('relation') != d['relation']
            or any(review.get(k)!=d.get(k) for k in ('source_sha256','target_term'))
            or d['target_type']!=term['term_type'] or d['source_locator']!=term['source_locator']
            or parse_attribute_constraints(review.get('required_attribute_constraints',[])) !=
               parse_attribute_constraints(d.get('required_attribute_constraints',[]))):
        raise ValueError('CEST review differs from its source-pinned policy')
    constraints=parse_attribute_constraints(d.get('required_attribute_constraints',[]))
    if ((d['relation']=='exact' and constraints) or (d['relation']=='compositional' and not constraints)
            or not cest_constraints_attested(context['bundles'].get('CEST'),term['term_id'],constraints)):
        raise ValueError('CEST review has unsupported source constraints')


CEST_REVIEW_VALIDATORS = {'CEST': validate_cest_applied_review}

# Insert after common source/evidence/rejection gates and before lexical fallback.
CEST_DECISION_RULES = [
    {'rule_id':'cest_target_not_pinned','when':{'terminology':'CEST','has_decision':False},
     'result':{'code':'cest_target_not_pinned','reason':'No source-pinned CEST decision exists for this class and source role.'}},
    {'rule_id':'cest_source_context_invalid','when':{'terminology':'CEST','cest_source_context_valid':False},
     'result':{'code':'cest_source_context_invalid','reason':'CEST source fingerprint or table-role locator differs from the imported source.'}},
    {'rule_id':'cest_exact_mapping_has_constraints','when':{'terminology':'CEST','decision_relation':'exact','has_constraints':True},
     'result':{'code':'exact_mapping_has_constraints','reason':'A constrained CEST expression cannot be an interchangeable unqualified term.'}},
    {'rule_id':'cest_composition_missing_constraints','when':{'terminology':'CEST','decision_relation':'compositional','has_constraints':False},
     'result':{'code':'composition_missing_constraints','reason':'A compositional CEST decision requires explicit source attributes.'}},
    {'rule_id':'cest_constraints_not_attested','when':{'terminology':'CEST','cest_constraints_attested':False},
     'result':{'code':'cest_constraints_not_attested','reason':'Every CEST qualifier must occur on this source term in the imported transcription.'}},
    {'rule_id':'cest_match_method_unsupported','when':{'terminology':'CEST','cest_match_supported':False},
     'result':{'code':'unsupported_match_method','reason':'CEST candidate lacks a source-located or exact lexical retrieval basis.'}},
    {'rule_id':'cest_source_pinned_evidence_decision','when':{'terminology':'CEST','decision_verdict':'approved'},
     'result':{'code':'cest_source_pinned_evidence_decision','verdict':'accepted'},
     'result_from':{'reason':'decision_rationale','relation':'decision_relation'}},
    {'rule_id':'cest_unresolved_decision','when':{'terminology':'CEST'},
     'result':{'code':'cest_unresolved_decision','reason':'CEST policy has no supported approval for this candidate.'}},
]
