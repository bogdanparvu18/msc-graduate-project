"""Exercise real S6 notebook functions without install, GPU, or workflow cells."""
import ast
import json
import re
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import pandas as pd


def load_temporal_namespace(notebook_path=None):
    """Load only the JSON helper definitions and the pure temporal policy cell."""
    if notebook_path is None:
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "phase2" / "phase2_06_temporal_context_images.ipynb"
        # Keep this helper runnable in the isolated review directory before integration.
        if not notebook_path.is_file():
            notebook_path = Path(__file__).with_name("s6_split_policy.ipynb")
    notebook_path = Path(notebook_path)
    notebook = json.loads(notebook_path.read_text())
    namespace = {"Path": Path, "PurePosixPath": PurePosixPath, "np": np, "pd": pd, "json": json, "re": re}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        if "def _s3_semantic_json_value(" in code:
            selected = [node for node in ast.parse(code).body
                        if isinstance(node, ast.FunctionDef)
                        and node.name in {"_s3_semantic_json_value", "_s3_semantic_json"}]
            exec(compile(ast.Module(body=selected, type_ignores=[]), str(notebook_path), "exec"), namespace)
        if "def enforce_temporal_split_isolation(" in code:
            exec(compile(code, str(notebook_path), "exec"), namespace)
    assert "enforce_temporal_split_isolation" in namespace, "S6 notebook lacks the split isolation policy"
    return namespace


def make_temporal_fixture(namespace, storage_root):
    """Return labelled/S4 inputs and safe, completed synthetic S6 tables.

    Two labelled frames cross splits as neighbouring contexts. One unlabelled
    physical frame is requested from two splits. All four targets remain valid.
    No image or video files are read or written.
    """
    ns = SimpleNamespace(**namespace)
    storage_root = Path(storage_root).resolve()
    config={'storage_root':str(storage_root),'temporal_context_enabled':True,
            'temporal_context_offsets_seconds':[-2,0,2], 'temporal_context_image_format':'jpg',
            'temporal_context_extract_frames':True}
    rows=[]
    for index,split,finding in [(10,'train','normal'),(12,'test','lesion'),(20,'train','normal'),(24,'test','lesion')]:
        rows.append({'annotation_id':f'a{index}','video_key':'v1','frame_number':index,
            'opencv_frame_index':index,'frame_index_offset':0,'split':split,
            'finding_class_normalized':finding,'image_path':str(storage_root / 'official' / f'{index}.jpg'),
            'image_relpath':f'official/{index}.jpg'})
    labelled=pd.DataFrame(rows)
    # Multiple official annotations remain intact and do not duplicate physical ownership.
    labelled=pd.concat([labelled,labelled.iloc[[0]].assign(annotation_id='a10-second-box')],ignore_index=True)
    original_labelled=labelled.copy(deep=True)
    assignments=labelled[['video_key','opencv_frame_index','split']].drop_duplicates().reset_index(drop=True)
    segments=pd.DataFrame([{
        'finding_segment_id':f's{r.frame_number}','video_key':r.video_key,
        'finding_class':r.finding_class_normalized,'finding_class_normalized':r.finding_class_normalized,
        'split':r.split,'target_frame_number':r.frame_number,'target_opencv_frame_index':r.opencv_frame_index,
        'target_image_annotation_id':r.annotation_id,'target_image_path':r.image_path,'target_image_relpath':r.image_relpath,
        'temporal_grounding_candidate':True,'temporal_supervision_level':'weak',
        'verified_temporal_boundary_available':False} for r in labelled.iloc[:4].itertuples()])
    videos=pd.DataFrame([{'video_key':'v1','video_path':str(storage_root / 'videos' / 'v1.mp4'),
        'video_relpath':'videos/v1.mp4','frame_count':100,'container_fps':1.0}])
    offsets=ns.build_verified_video_offsets(labelled)
    raw=ns.build_temporal_manifest(segments,videos,config,offsets)
    enriched=ns.add_temporal_context_metadata(raw,labelled)
    original_enriched=enriched.copy(deep=True)
    safe=ns.enforce_temporal_split_isolation(enriched,labelled,assignments)
    requests=ns.build_temporal_extraction_requests(safe,videos,{'temporal_frames_dir':str(storage_root / 'derived')},config)
    results=ns.s6_typed_table(pd.DataFrame([{
        'video_key':r.video_key,'opencv_frame_index':r.opencv_frame_index,
        'extracted_image_path':r.output_path,'extracted_image_relpath':r.output_relpath,
        'extraction_status':'success','extraction_error':pd.NA,'source_video_sha256':'1'*64,
        'extracted_image_sha256':'2'*64,'decoder_backend':'test'} for r in requests.itertuples()],
        columns=ns.TEMPORAL_EXTRACTION_RESULT_COLUMNS))
    attached=ns.attach_temporal_extraction_results(safe,results)
    audit=attached.loc[attached.context_image_path.notna(),['context_image_path','context_image_relpath']].drop_duplicates()
    audit=audit.assign(image_readable=True,image_sha256='3'*64,image_width=336,image_height=336,image_validation_error=pd.NA)
    audit=ns.s6_typed_table(audit[ns.S6_AUDIT_COLUMNS])
    complete=ns.attach_temporal_image_audit(attached,audit)
    return {'labelled':labelled,'original_labelled':original_labelled,'assignments':assignments,
            'safe':safe,'complete':complete,'raw':raw,'enriched':enriched,'original_enriched':original_enriched,
            'config':config,'videos':videos,'segments':segments,'requests':requests,'results':results,
            'attached':attached,'audit':audit}


def test_temporal_split_isolation(tmp_path):
    namespace = load_temporal_namespace()
    ns = SimpleNamespace(**namespace)
    storage_root = tmp_path.resolve()
    fixture = make_temporal_fixture(namespace, storage_root)
    labelled = fixture["labelled"]
    original_labelled = fixture["original_labelled"]
    assignments = fixture["assignments"]
    safe = fixture["safe"]
    raw = fixture["raw"]
    enriched = fixture["enriched"]
    original_enriched = fixture["original_enriched"]
    config = fixture["config"]
    videos = fixture["videos"]
    segments = fixture["segments"]
    assert len(safe)==len(raw)==12
    pd.testing.assert_frame_equal(labelled,original_labelled)
    pd.testing.assert_frame_equal(enriched,original_enriched)
    assert safe.loc[safe.is_target,'context_split_allowed'].all()
    assert safe.context_split_allowed.sum()==8
    assert safe.context_split_status.value_counts().to_dict()=={
        'allowed_labelled_split':4,'allowed_unlabelled_single_split':4,
        'excluded_labelled_split_mismatch':2,'excluded_unlabelled_cross_split':2}
    assert safe.loc[safe.opencv_frame_index.eq(22),'context_image_path'].isna().all()
    # Labelled split owner retains the image; foreign context does not.
    assert safe.loc[safe.opencv_frame_index.eq(10)&safe.split.eq('train'),'context_split_allowed'].all()
    assert not safe.loc[safe.opencv_frame_index.eq(10)&safe.split.eq('test'),'context_split_allowed'].any()
    blocked=safe.loc[~safe.context_split_allowed]
    assert blocked[['context_image_path','context_image_relpath']].isna().all().all()
    active_columns=[c for c in safe if c.startswith(('context_official_','context_verified_',
                   'context_image_annotation','context_image_verified'))]
    assert blocked[active_columns].isna().all().all()
    assert not blocked.context_is_verified.any() and not blocked.context_is_verified_same_finding.any()
    for _,row in blocked.iterrows():
        audit=json.loads(row.audit_excluded_context_metadata_json)
        assert audit['context_annotation_status']!='excluded_split_isolation'
        assert row.context_annotation_status=='excluded_split_isolation'
        if row.context_frame_split is not pd.NA and pd.notna(row.context_frame_split):
            assert audit['context_official_image_path'] is not None
    requests=ns.build_temporal_extraction_requests(safe,videos,{'temporal_frames_dir':str(storage_root / 'derived')},config)
    assert set(requests.opencv_frame_index)=={8,14,18,26}
    results=ns.s6_typed_table(pd.DataFrame([{
        'video_key':r.video_key,'opencv_frame_index':r.opencv_frame_index,
        'extracted_image_path':r.output_path,'extracted_image_relpath':r.output_relpath,
        'extraction_status':'success','extraction_error':pd.NA,'source_video_sha256':'1'*64,
        'extracted_image_sha256':'2'*64,'decoder_backend':'test'} for r in requests.itertuples()],
        columns=ns.TEMPORAL_EXTRACTION_RESULT_COLUMNS))
    assert ns.validate_temporal_result_coverage(requests,results)
    # Even a mistakenly supplied successful shared-frame result cannot reactivate blocked rows.
    extra=results.iloc[[0]].copy();extra['opencv_frame_index']=22
    extra['extracted_image_path']=str(storage_root / 'derived' / 'v1' / 'frame_00000022.jpg')
    extra['extracted_image_relpath']='derived/v1/frame_00000022.jpg'
    attached=ns.attach_temporal_extraction_results(safe,pd.concat([results,extra],ignore_index=True))
    assert attached.loc[~attached.context_split_allowed,['context_image_path','context_image_relpath',
        'extracted_image_path','extracted_image_relpath','extracted_image_sha256']].isna().all().all()
    assert attached.loc[~attached.context_split_allowed,'extraction_status'].eq('blocked_split_isolation').all()
    audit=attached.loc[attached.context_image_path.notna(),['context_image_path','context_image_relpath']].drop_duplicates()
    audit=audit.assign(image_readable=True,image_sha256='3'*64,image_width=336,image_height=336,image_validation_error=pd.NA)
    audit=ns.s6_typed_table(audit[ns.S6_AUDIT_COLUMNS])
    complete=ns.attach_temporal_image_audit(attached,audit)
    assert not complete.loc[~complete.context_split_allowed,'image_readable'].any()
    assert ns.build_temporal_extraction_failures(complete,config).empty
    windows=ns.build_temporal_window_report(complete)
    assert windows.requested_context_count.eq(3).all()
    assert windows.available_context_count.eq(3).all()
    assert windows.temporal_window_complete.all()
    assert windows.split_allowed_context_count.eq(2).all()
    assert windows.split_excluded_context_count.eq(1).all()
    assert windows.usable_context_count.eq(2).all()
    assert not windows.temporal_window_usable_complete.any()
    status=ns.build_temporal_status_report(complete)
    assert status.loc[~status.context_split_allowed,'row_count'].sum()==4
    # Independent assignment/target validation.
    def expect_error(call,phrase):
        try:call()
        except ValueError as error:assert phrase in str(error),str(error)
        else:raise AssertionError('Expected ValueError')
    bad=assignments.copy();bad.loc[0,'split']='validation'
    expect_error(lambda:ns.enforce_temporal_split_isolation(enriched,labelled,bad),'disagree')
    expect_error(lambda:ns.enforce_temporal_split_isolation(enriched,labelled,pd.concat([assignments,assignments.iloc[[0]]])),'one row')
    bad_target=enriched.copy();bad_target.loc[bad_target.opencv_frame_index.eq(8),'is_target']=True
    expect_error(lambda:ns.enforce_temporal_split_isolation(bad_target,labelled,assignments),'Every temporal target')
    shuffled=ns.enforce_temporal_split_isolation(enriched.sample(frac=1,random_state=7),labelled,assignments)
    keys=['finding_segment_id','opencv_frame_index']
    pd.testing.assert_frame_equal(safe.sort_values(keys).reset_index(drop=True),shuffled.sort_values(keys).reset_index(drop=True))
    # Empty temporal candidate case retains typed report schemas.
    empty=ns.enforce_temporal_split_isolation(enriched.iloc[:0],labelled,assignments)
    empty=ns.attach_temporal_extraction_results(empty,ns.s6_empty_table(ns.TEMPORAL_EXTRACTION_RESULT_COLUMNS))
    empty=ns.attach_temporal_image_audit(empty,ns.s6_empty_table(ns.S6_AUDIT_COLUMNS))
    assert ns.build_temporal_window_report(empty).empty
    assert ns.build_temporal_extraction_failures(empty,config).empty
    # Shared unlabelled context within a single split remains eligible, extracted once.
    same_split=enriched.loc[enriched.opencv_frame_index.eq(8)].copy()
    same_split=pd.concat([same_split,same_split.assign(finding_segment_id='same-split-other-segment')],ignore_index=True)
    same_split=ns.enforce_temporal_split_isolation(same_split,labelled,assignments)
    assert same_split.context_split_allowed.all()
    same_requests=ns.build_temporal_extraction_requests(same_split,videos,{'temporal_frames_dir':str(storage_root / 'derived')},config)
    assert len(same_requests)==1 and same_requests.opencv_frame_index.iloc[0]==8
    # Fully blocked context-only subset: no extraction or false failure rows.
    only_blocked=enriched.loc[~safe.context_split_allowed].copy()
    only_blocked=ns.enforce_temporal_split_isolation(only_blocked,labelled,assignments)
    assert not only_blocked.context_split_allowed.any()
    no_requests=ns.build_temporal_extraction_requests(only_blocked,videos,{'temporal_frames_dir':str(storage_root / 'derived')},config)
    assert no_requests.empty
    only_blocked=ns.attach_temporal_extraction_results(only_blocked,ns.s6_empty_table(ns.TEMPORAL_EXTRACTION_RESULT_COLUMNS))
    only_blocked=ns.attach_temporal_image_audit(only_blocked,ns.s6_empty_table(ns.S6_AUDIT_COLUMNS))
    assert ns.build_temporal_extraction_failures(only_blocked,config).empty
    blocked_windows=ns.build_temporal_window_report(only_blocked)
    assert blocked_windows.split_allowed_context_count.eq(0).all()
    assert blocked_windows.usable_context_count.eq(0).all()
    # A real allowed-image failure remains a failure, distinct from intentional exclusions.
    failed_audit=audit.copy();failed_audit.loc[failed_audit.index[0],'image_readable']=False
    failed_audit.loc[failed_audit.index[0],'image_validation_error']='test_decode_failure'
    partial=ns.attach_temporal_image_audit(attached,failed_audit)
    failures=ns.build_temporal_extraction_failures(partial,config)
    assert len(failures)==1 and failures.image_validation_error.eq('test_decode_failure').all()
    summary=ns.build_sector6_summary({'temporal_manifest':complete,'temporal_extraction_requests':requests,
        'extracted_temporal_frames':results,'temporal_window_report':windows,
        'temporal_extraction_failures':ns.build_temporal_extraction_failures(complete,config)})
    assert summary['split_excluded_context_rows']==4
    assert summary['split_allowed_context_rows']==8
    assert summary['missing_or_unreadable_context_rows']==0
    assert summary['validation_passed'] is True and summary['execution_outcome']=='complete'
