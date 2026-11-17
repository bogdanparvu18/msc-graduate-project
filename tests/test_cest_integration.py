"""Offline CEST tests: actual notebook functions, no Colab, RF2 or clinical claims."""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timezone
from urllib.parse import quote, unquote, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen, HTTPRedirectHandler, build_opener
from typing import Any, Callable, Iterable, Mapping, Sequence
import ast
import copy
import csv
import fnmatch
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
import zipfile
import pytest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase3/sector1/terminology'
NB=ROOT/'notebooks/phase3/phase3_01_terminology_mapping.ipynb'

def code(index):
    return ''.join(json.loads(NB.read_text())['cells'][index]['source'])

def exec_source(source,ns):
    exec(compile('from __future__ import annotations\n'+source,'<notebook-test>','exec'),ns)

@pytest.fixture(scope='module')
def engine():
    ns=dict(globals(),IntegrityError=ValueError,_BLOCK_SIZE=1024*1024,
            _SHA256=re.compile(r'[0-9a-f]{64}\Z'),_COMMIT=re.compile(r'[0-9a-f]{40}\Z'),
            TERMINOLOGY_CONFIG={'mapping_review_path':None})
    wanted={'canonical_json','canonical_sha256','parse_attribute_constraints','clean_json_value',
            '_rag_github_path','_rag_github_sha','_rag_github_json','_rag_github_local_path',
            '_rag_github_temporary','validate_github_manifest','load_github_verified_artifact','file_digest','verify_file'}
    nodes=[n for n in ast.parse(code(3)).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    exec_source(ast.unparse(ast.Module(body=nodes,type_ignores=[])),ns)
    ns['GITHUB_MANIFEST_LIMITS'] = {'manifest_bytes':1024*1024,'artifact_bytes':32*1024*1024}
    exec_source(code(11).split('GI_GITHUB_CONFIG =')[0],ns)
    # SNOMED policy validation is used, but no actual RF2 or network is loaded.
    nodes=[n for n in ast.parse(code(13)).body if isinstance(n,ast.FunctionDef) and n.name=='snomed_description_policy']
    exec_source(ast.unparse(ast.Module(body=nodes,type_ignores=[])),ns)
    exec_source(code(15).split('if IN_COLAB and CONFIG["mount_google_drive"]:')[0],ns)
    exec_source(code(18).split('# Reuse section 1')[0],ns)
    return ns

@pytest.fixture
def assets(tmp_path,engine):
    package=json.loads((OUT/'cest_2005_package.json').read_text())
    # An explicit synthetic document tests byte identity without redistributing the licensed PDF in CI.
    doc=tmp_path/'CEST.pdf';doc.write_bytes(b'%PDF-1.4\nsynthetic integrity fixture, not clinical source\n%%EOF\n')
    package['metadata']['source_sha256']=hashlib.sha256(doc.read_bytes()).hexdigest()
    package['metadata']['source_bytes']=doc.stat().st_size
    path=tmp_path/'cest.json';path.write_text(json.dumps(package,ensure_ascii=False))
    bundle=engine['read_cest_package'](path,doc)
    rag=json.loads((OUT/'phase3_01_rag_evidence.json').read_text())
    for d in rag['target_decisions']:
        if d['terminology']=='CEST':d['source_sha256']=bundle['manifest']['source_fingerprint']
    classes=pd.DataFrame([{'dataset_class_id':'test:'+p['class_label'],'normalized_label':p['class_label']} for p in rag['profiles']])
    return dict(path=path,doc=doc,package=package,bundle=bundle,rag=rag,classes=classes)


def prepared(engine,a):
    return engine['propose_mapping_candidates'](a['classes'],{'CEST':a['bundle']},{'rows':[]},evidence_bundle=a['rag'])


def test_published_manifest(engine):
    raw=(OUT/'cest_2005_manifest.json').read_bytes()
    m=engine['_cest_github_manifest'](raw)
    p=engine['_cest_github_package']((OUT/'cest_2005_package.json').read_bytes(),m)
    assert p['metadata']['source_file']=='CEST.pdf'
    assert p['metadata']['human_reviewed'] is False
    assert len(p['terms'])==128 and len(p['attributes'])==395
    assert {r['table'] for r in p['source_contexts']}==set(range(1,12))


def test_actual_authorized_pdf(engine):
    pdf=Path(os.environ.get('CEST_TEST_PDF','/mnt/data/CEST.pdf'))
    if not pdf.is_file():pytest.skip('Original authorized PDF is not distributed with CI')
    b=engine['read_cest_package'](OUT/'cest_2005_package.json',pdf)
    m=json.loads((OUT/'cest_2005_manifest.json').read_text())
    assert b['manifest']['source_fingerprint']==m['source_fingerprint']


def test_policy_hashes(engine):
    raw=(OUT/'phase3_01_rag_evidence.json').read_bytes();r=json.loads(raw)
    m=json.loads((OUT/'rag_evidence_manifest.json').read_text())
    assert len(raw)==m['artifact']['bytes']
    assert hashlib.sha256(raw).hexdigest()==m['artifact']['sha256']
    assert engine['canonical_sha256'](r)==m['bundle_content_sha256']
    assert engine['rag_validate_bundle'](r)[-1]==m['bundle_content_sha256']
    f=json.loads((OUT/'ulcer_policy_regression.json').read_text())
    assert f['policy_reference']['bundle_content_sha256']==m['bundle_content_sha256']


def test_end_to_end_project_mapping(engine,assets):
    p=prepared(engine,assets);r=engine['build_rag_mapping_reviews'](p,assets['rag'])
    out=engine['apply_mapping_decisions'](p,r['reviews'])
    rows=out['crosswalk'].query('terminology == "CEST"')
    assert rows.mapping_status.value_counts().to_dict()=={'approved':16,'unmapped':2}
    approved=rows.query('mapping_status == "approved"')
    assert approved.relation.value_counts().to_dict()=={'related':8,'exact':7,'compositional':1}
    aliases=out['approved_class_synonyms']; assert len(aliases)==7
    assert not aliases['dataset_class_id'].str.contains('blood|erythema|lymphangiectasia').any()
    audit=engine['build_source_synonym_audit'](out,assets['rag'])
    assert audit['rag_decision'].eq('accepted_source_synonym').all()


@pytest.mark.parametrize('name',['Ulcer','Erosion','Angiectasia'])
def test_same_spelling_distinct_roles(assets,name):
    terms=assets['bundle']['terms'];rows=terms[terms.preferred_term.eq(name)]
    assert set(rows.term_type)=={'finding','diagnosis'} and rows.term_id.nunique()==2


def test_categorical_size_and_blank_values(assets):
    p=assets['package'];ids={r['preferred_term']:r['term_id'] for r in p['terms'] if r['term_type']=='finding'}
    size=[a for a in p['attributes'] if a['term_id']==ids['Polyp(s)'] and a['attribute']=='Size']
    assert {a['value'] for a in size}=={'Small','Medium','Large'}
    assert all(a['value_kind']=='enumerated' for a in size)
    blank=[a for a in p['attributes'] if a['term_id']==ids['Foreign Body']]
    assert [(a['attribute'],a['value'],a['value_kind']) for a in blank]==[('Type','','unspecified_in_source')]
    assert p['synonyms']==[]
    assert not any(a['term_id']==ids['Diverticulum'] for a in p['attributes'])
    assert any(u['term_id']==ids['Diverticulum'] for u in p['unbound_source_values'])


def test_cross_column_continuations(assets):
    a=assets['bundle']['attributes']
    for tid in ['CEST2005:T06:angiectasia','CEST2005:T06:venous_structure']:
        d=a[a.term_id.eq(tid)&a.attribute.eq('Distribution pattern')]
        assert set(d.value)=={'Localized','Patchy','Diffuse'}
        assert d.source_locator.str.contains('continuation').all()


@pytest.mark.parametrize('field,value',[
    ('source_sha256','0'*64),('source_bytes',12),('human_reviewed','false'),('transcription_reviewed_by',''),
])
def test_bad_metadata_rejected(engine,assets,field,value):
    p=assets['package'];p['metadata'][field]=value;assets['path'].write_text(json.dumps(p))
    with pytest.raises(ValueError):engine['read_cest_package'](assets['path'],assets['doc'])


@pytest.mark.parametrize('mutation',['term_duplicate','context_duplicate','bad_parent','bad_value_kind','unknown_attribute_target','invented_synonym'])
def test_bad_transcription_rejected(engine,assets,mutation):
    p=assets['package']
    if mutation=='term_duplicate':p['terms'].append(copy.deepcopy(p['terms'][0]))
    if mutation=='context_duplicate':p['source_contexts'][1]=p['source_contexts'][0]
    if mutation=='bad_parent':p['source_contexts'][0]['parent_term_id']='unknown'
    if mutation=='bad_value_kind':p['attributes'][0]['value_kind']='guessed_measurement'
    if mutation=='unknown_attribute_target':p['attributes'][0]['term_id']='unknown'
    if mutation=='invented_synonym':p['synonyms']=[{'term_id':p['terms'][0]['term_id'],'synonym':'guess','relation':'similar'}]
    with pytest.raises(ValueError):engine['validate_cest_transcription'](p)


def test_tampered_original_rejected(engine,assets):
    assets['doc'].write_bytes(assets['doc'].read_bytes().replace(b'synthetic',b'malformed'))
    with pytest.raises(ValueError):engine['read_cest_package'](assets['path'],assets['doc'])


def test_transcription_edit_changes_pair_fingerprint(engine,assets):
    p=assets['package'];p['metadata']['transcription_reviewed_by']+=' revision'
    assets['path'].write_text(json.dumps(p))
    b=engine['read_cest_package'](assets['path'],assets['doc'])
    assert b['manifest']['source_fingerprint']!=assets['bundle']['manifest']['source_fingerprint']


@pytest.mark.parametrize('bad', ['hash','target','locator','attribute'])
def test_bad_pinned_proposal_is_audited(engine,assets,bad):
    d=next(d for d in assets['rag']['target_decisions'] if d['terminology']=='CEST' and d['class_label']=='blood hematin')
    if bad=='hash':d['source_sha256']='0'*64
    if bad=='target':d['target_id']='CEST2005:missing'
    if bad=='locator':d['source_locator']='wrong table'
    if bad=='attribute':d['required_attribute_constraints']=[{'attribute':'Type','value':'Fresh'}]
    p=prepared(engine,assets);a=p['cest_proposal_audit'].query('class_label == "blood hematin"')
    assert len(a)==1 and a.iloc[0]['status']!='candidate_proposed'
    r=engine['build_rag_mapping_reviews'](p,assets['rag'])
    assert not any(x['class_label']=='blood hematin' and x['terminology']=='CEST' for x in r['reviews']['reviews'])


def test_absent_source_stays_source_required(engine,assets):
    # Use an empty non-CEST source to meet the preexisting candidate input contract.
    b={k:v.iloc[:0] if isinstance(v,pd.DataFrame) else v for k,v in assets['bundle'].items()}
    b['manifest']={'sha256':'0'*64}
    p=engine['propose_mapping_candidates'](assets['classes'],{'MST':b},{'rows':[]},evidence_bundle=assets['rag'])
    assert p['crosswalk'].query('terminology == "CEST"').mapping_status.eq('source_required').all()


@pytest.mark.parametrize('bad',['relation','constraint','unqualified_hematin','wrong_role'])
def test_final_gate_cannot_be_bypassed(engine,assets,bad):
    p=prepared(engine,assets);r=engine['build_rag_mapping_reviews'](p,assets['rag'])
    rev=copy.deepcopy(r['reviews'])
    d=next(d for d in rev['reviews'] if d['terminology']=='CEST' and d['class_label']=='blood hematin')
    if bad=='relation':d['relation']='exact'
    if bad=='constraint':d['required_attribute_constraints']=[{'attribute':'Type','value':'Fresh'}]
    if bad=='unqualified_hematin':d['relation']='related';d['required_attribute_constraints']=[]
    if bad=='wrong_role':d['target_id']='CEST2005:T10:ulcer';d['target_term']='Ulcer'
    with pytest.raises(ValueError):engine['apply_mapping_decisions'](p,rev)


@pytest.mark.parametrize('gate',['evidence','fingerprint','role','unpinned'])
def test_common_gates_precede_cest_acceptance(engine,assets,gate):
    p=prepared(engine,assets)
    if gate=='evidence':
        for s in assets['rag']['sources']:s['independent_of_dataset']=False
    elif gate=='fingerprint':
        for d in assets['rag']['target_decisions']:
            if d['terminology']=='CEST':d['source_sha256']='0'*64
    elif gate=='role':
        for d in assets['rag']['target_decisions']:
            if d['terminology']=='CEST':d['target_type']='invented'
    else:
        assets['rag']['target_decisions']=[d for d in assets['rag']['target_decisions'] if d['terminology']!='CEST']
    r=engine['build_rag_mapping_reviews'](p,assets['rag'])
    assert not any(d['terminology']=='CEST' for d in r['reviews']['reviews'])


def test_rules_remain_declarative_and_ordered(engine):
    pipeline=engine['RAG_DECISION_PIPELINE'];engine['rag_validate_decision_pipeline'](pipeline,engine['RAG_DECISION_ACTIONS'])
    json.dumps(pipeline)
    names=[r['rule_id'] for r in pipeline]
    assert names.index('explicit_evidence_rejection')<names.index('cest_target_not_pinned')<names.index('unsupported_terminology')
    assert len(pipeline)==40


def test_cest_discovery_only_named_pdf(engine,tmp_path):
    (tmp_path/'CEST.pdf').write_bytes(b'x');(tmp_path/'cest_source.pdf').write_bytes(b'y')
    assert engine['terminology_file_choice'](None,tmp_path,['CEST.pdf']).name=='CEST.pdf'
    assert '"cest_document_patterns": ["CEST.pdf"]' in code(3)
    assert 'cest_source.pdf' not in code(15)


def test_verified_loader_reuses_cache(engine,tmp_path):
    calls=[]
    def fetch(url,destination):
        calls.append(url)
        source=OUT/('cest_2005_manifest.json' if url.endswith('manifest.json') else 'cest_2005_package.json')
        Path(destination).write_bytes(source.read_bytes())
    args=dict(repository='test/repo',ref='main',manifest_path='outputs/phase3/sector1/terminology/cest_2005_manifest.json',
              cache_root=tmp_path,fetch=fetch,attempts=1)
    a,p=engine['load_github_cest_package'](**args);b,q=engine['load_github_cest_package'](**args)
    assert a==b and q['cache_status']=='reused_verified'
    assert sum(u.endswith('cest_2005_package.json') for u in calls)==1


def test_verified_loader_rejects_tampering(engine,tmp_path):
    def fetch(url,destination):
        source=OUT/('cest_2005_manifest.json' if url.endswith('manifest.json') else 'cest_2005_package.json')
        Path(destination).write_bytes(source.read_bytes()+(b' ' if source.name.endswith('package.json') else b''))
    with pytest.raises(RuntimeError):engine['load_github_cest_package']('test/repo','main','cest_2005_manifest.json',tmp_path,fetch,1)


def test_notebook_code_compiles():
    for c in json.loads(NB.read_text())['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))


def test_core_contract_cells_unchanged():
    expected=json.loads((Path(__file__).with_name('cest_regression_baseline.json')).read_text())
    for i,digest in expected['unchanged_cell_sources'].items():
        assert hashlib.sha256(code(int(i)).encode()).hexdigest()==digest
    r=json.loads((OUT/'phase3_01_rag_evidence.json').read_text())
    canonical=lambda v:hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert canonical(r['profiles'])==expected['profiles_sha256']
    assert canonical([d for d in r['target_decisions'] if d['terminology']!='CEST'])==expected['non_cest_decisions_sha256']


def test_891_coverage_decisions_unchanged(engine):
    s=code(9);module=ast.parse(s);end=next(n.end_lineno for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_coverage_expectation')
    ns={};exec_source('\n'.join(s.splitlines()[:end]),ns)
    rows=[ns['_coverage_expectation'](t,v,e) for t in ns['TASK_SPECIFICATION'] for v in ns['VARIANT_SPECIFICATION'] for e in ns['EVIDENCE_STATES']]
    assert len(rows)==891
    expected=json.loads((Path(__file__).with_name('cest_regression_baseline.json')).read_text())
    assert engine['canonical_sha256'](rows)==expected['coverage_sha256']


def test_source_tables_parquet_roundtrip(assets,tmp_path):
    pytest.importorskip('pyarrow')
    for name in ['terms','attributes','source_contexts','source_notes','unbound_source_values']:
        table=assets['bundle'][name];p=tmp_path/(name+'.parquet');table.to_parquet(p,index=False)
        assert pd.read_parquet(p).shape==table.shape


def test_cest_reuses_shared_manifest_validator(engine,monkeypatch):
    original=engine["validate_github_manifest"]; calls=[]
    def traced(data,specification):
        calls.append(specification["label"])
        return original(data,specification)
    monkeypatch.setitem(engine,"validate_github_manifest",traced)
    engine["_cest_github_manifest"]((OUT/"cest_2005_manifest.json").read_bytes())
    assert calls==["CEST"]


@pytest.mark.parametrize("change",["schema","oversized"])
def test_cest_shared_manifest_guards(engine,change):
    manifest=json.loads((OUT/"cest_2005_manifest.json").read_text())
    if change=="schema":manifest["artifact"]["schema_version"]="wrong"
    data=json.dumps(manifest).encode()
    if change=="oversized":data=b" "*(1024*1024+1)+data
    with pytest.raises(ValueError):engine["_cest_github_manifest"](data)
