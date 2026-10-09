"""Check the real S6 -> S7 boundary and CPU QC without Drive or CUDA."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from test_sector7_recovery import ns
from test_temporal_split_isolation import load_temporal_namespace, make_temporal_fixture


def test_split_safe_s6_references_survive_real_sector7_qc(ns, tmp_path):
    s6 = load_temporal_namespace()
    fixture = make_temporal_fixture(s6, tmp_path)
    temporal, labelled = fixture['complete'], fixture['labelled']
    assert ns['validate_sector7_temporal_split_contract'](temporal, fixture['assignments'])
    config = {**ns['S7_DEFAULTS'], 'storage_root': str(tmp_path)}
    inventory = ns['build_qc_image_inventory'](labelled, temporal, config)
    assert len(inventory) == 8
    assert inventory['image_relpath'].is_unique

    # Generate actual compressed images for CPU fallback and content auditing.
    pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    for filename in inventory['image_path']:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), pixels)
    audit6 = fixture['audit'].copy()
    audit6['image_sha256'] = audit6['context_image_path'].map(ns['_p3_file_hash']).astype('string')
    temporal = s6['attach_temporal_image_audit'](fixture['attached'], audit6)
    source_audit = ns['io_audit_sector7_sources'](inventory, audit6, config)
    inputs = {'sector5': {'labelled_frame_manifest_with_segments': labelled},
              'sector6': {'temporal_manifest': temporal, 'temporal_image_audit': audit6}}
    with pytest.warns(UserWarning) as warnings:
        tables = ns['io_compute_sector7_tables'](inputs, config, inventory, source_audit)
    assert len(warnings) == 1  # One CPU fallback warning for this complete run.
    assert len(tables) == 10
    labelled_qc = tables['labelled_frame_manifest_with_qc']
    temporal_qc = tables['temporal_manifest_with_qc']
    pd.testing.assert_frame_equal(labelled_qc[labelled.columns], labelled.reset_index(drop=True))
    pd.testing.assert_frame_equal(temporal_qc[temporal.columns], temporal.reset_index(drop=True))
    excluded = ~temporal_qc['context_split_allowed']
    assert excluded.sum() == 4
    assert temporal_qc.loc[excluded, 'qc_requires_review'].all()
    assert not temporal_qc.loc[excluded, 'image_usable_for_modelling'].any()
    assert not temporal_qc.loc[excluded, 'qc_reference_covered'].any()
    assert temporal_qc.loc[~excluded, 'image_usable_for_modelling'].all()
    report = tables['missing_image_reference_report']
    assert len(report) == 4
    assert report['reference_status'].eq('excluded_split_isolation').all()
    assert tables['image_qc_thresholds'].iloc[0]['calibration_candidate_images'] == 4
    assert tables['image_qc_manifest']['qc_source_matches_sector6'].all()


@pytest.mark.parametrize('mutation', ['legacy', 'allow_flag', 'owner', 'path', 'label', 'extracted'])
def test_sector7_rejects_legacy_or_inconsistent_temporal_evidence(ns, tmp_path, mutation):
    fixture = make_temporal_fixture(load_temporal_namespace(), tmp_path)
    temporal = fixture['complete'].copy()
    row = temporal.index[~temporal['context_split_allowed']][0]
    if mutation == 'legacy':
        temporal = temporal.drop(columns='context_split_allowed')
    elif mutation == 'allow_flag':
        temporal.loc[row, 'context_split_allowed'] = True
    elif mutation == 'owner':
        temporal.loc[row, 'context_frame_split'] = temporal.loc[row, 'split']
    elif mutation == 'path':
        temporal.loc[row, 'context_image_path'] = str(tmp_path / 'foreign.jpg')
    elif mutation == 'label':
        temporal.loc[row, 'context_verified_labels'] = 'foreign_ground_truth'
    else:
        temporal.loc[row, 'extracted_image_path'] = str(tmp_path / 'foreign.jpg')
    with pytest.raises(RuntimeError):
        ns['validate_sector7_temporal_split_contract'](temporal, fixture['assignments'])


def test_sector7_accepts_empty_split_checked_temporal_manifest(ns, tmp_path):
    fixture = make_temporal_fixture(load_temporal_namespace(), tmp_path)
    assert ns['validate_sector7_temporal_split_contract'](
        fixture['complete'].iloc[:0], fixture['assignments'])


def test_publication_is_explicitly_disabled_in_sector7(ns):
    assert ns['SECTOR7_SETTINGS']['push'] is False
    assert ns['SECTOR7_SETTINGS']['dry_run'] is False
    assert ns['io_push_sector7']({}, {'github': {'push': False, 'dry_run': False}})['status'] == 'not_requested'
