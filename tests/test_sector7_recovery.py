"""Regression checks for notebook-local cache and image read-error recovery."""
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NOTEBOOK = Path(os.environ.get('S7_TEST_NOTEBOOK', Path(__file__).resolve().parents[1] /
    'notebooks/phase2/phase2_07_image_quality_control.ipynb'))

@pytest.fixture
def ns():
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {}
    for i in (4, 6, 8, 10, 12, 14):
        tree = ast.parse(''.join(notebook['cells'][i]['source']))
        tree.body = [node for node in tree.body
                     if not (isinstance(node, ast.ImportFrom) and node.module == 'IPython.display')]
        exec(compile(tree, f'{NOTEBOOK.name}:cell{i}', 'exec'), namespace)
    namespace['cp'] = namespace['nvimgcodec'] = None
    return namespace


def sample(ns, tmp_path, monkeypatch=None, inaccessible=False):
    import cv2
    image = tmp_path / 'sample.jpg'
    cv2.imwrite(str(image), np.full((16, 16, 3), 128, dtype=np.uint8))
    if inaccessible:
        real_hash, real_read_bytes = ns['_p3_file_hash'], Path.read_bytes
        def inaccessible_hash(path):
            if Path(path) == image:
                raise PermissionError(13, 'Permission denied', str(image))
            return real_hash(path)
        def inaccessible_bytes(path):
            if path == image:
                raise PermissionError(13, 'Permission denied', str(image))
            return real_read_bytes(path)
        monkeypatch.setitem(ns, '_p3_file_hash', inaccessible_hash)
        monkeypatch.setattr(Path, 'read_bytes', inaccessible_bytes)
    upstream = tmp_path / 'upstream.json'
    upstream.write_text('{}')
    config = {**ns['S7_DEFAULTS'], 'phase': 2, 'sector': 7,
              'storage_root': str(tmp_path),
              'sector7': {'local_output_dir': 'outputs/phase2/sector7', 'push': False}}
    labelled = pd.DataFrame({'image_path': [str(image)], 'image_relpath': ['sample.jpg'],
                             'split': ['train'], 'annotation_id': ['a']})
    temporal = pd.DataFrame({'context_image_path': pd.Series(dtype='string'),
                             'context_image_relpath': pd.Series(dtype='string'),
                             'split': pd.Series(dtype='string'),
                             'context_image_source': pd.Series(dtype='string')})
    audit6 = pd.DataFrame({'context_image_relpath': pd.Series(dtype='string'),
                          'image_sha256': pd.Series(dtype='string')})
    inventory = ns['build_qc_image_inventory'](labelled, temporal, config)
    audit = ns['io_audit_sector7_sources'](inventory, audit6, config)
    inputs = {'sector5': {'labelled_frame_manifest_with_segments': labelled},
              'sector6': {'temporal_manifest': temporal, 'temporal_image_audit': audit6}}
    with pytest.warns(UserWarning):
        tables = ns['io_compute_sector7_tables'](inputs, config, inventory, audit)
    lineage = {'input_sha256': {'upstream': ns['_p3_file_hash'](upstream)},
               'input_descriptors': {'upstream': str(upstream)},
               'core_source_sha256': 'a' * 64,
               'image_source_audit_sha256': ns['sector3_table_fingerprint'](audit)}
    return image, upstream, config, tables, lineage


def test_unchanged_permission_failure_is_saved_as_unreadable(ns, tmp_path, monkeypatch):
    _, _, config, tables, lineage = sample(ns, tmp_path, monkeypatch, inaccessible=True)
    assert tables['image_qc_source_audit']['source_status'].iloc[0] == 'unreadable'
    assert len(tables['image_read_failure_report']) == 1
    assert not tables['image_qc_manifest']['image_readable'].iloc[0]
    assert not tables['image_qc_manifest']['image_usable_for_modelling'].iloc[0]
    saved = ns['io_persist_sector7'](tables, config, lineage)
    assert ns['_p7_read_state'](saved['root']) == saved['state']
    assert ns['io_persist_sector7'](tables, config, lineage)['status'] == 'unchanged'


def test_unchanged_permission_failure_remains_reportable(ns, tmp_path, monkeypatch):
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'unchanged')
    real_hash = ns['_p3_file_hash']
    def inaccessible(path):
        if Path(path) == image:
            raise PermissionError(13, 'Permission denied', str(path))
        return real_hash(path)
    monkeypatch.setitem(ns, '_p3_file_hash', inaccessible)
    current = ns['io_inspect_sector7_source'](image)
    assert current['source_status'] == 'unreadable'
    assert pd.isna(current['image_sha256'])
    audit = pd.DataFrame([{'image_path': str(image), 'image_relpath': image.name, **current}])
    assert ns['io_validate_sector7_source_audit'](audit)
    monkeypatch.setitem(ns, '_p3_file_hash', real_hash)
    with pytest.raises(RuntimeError, match='changed during'):
        ns['io_validate_sector7_source_audit'](audit)


@pytest.mark.parametrize('change', ['bytes', 'disappear', 'appear', 'error_kind'])
def test_source_changes_still_rejected(ns, tmp_path, monkeypatch, change):
    image = tmp_path / 'image.jpg'
    if change != 'appear':
        image.write_bytes(b'one')
    if change == 'error_kind':
        monkeypatch.setitem(ns, '_p3_file_hash', lambda path: (_ for _ in ()).throw(PermissionError('denied')))
    current = ns['io_inspect_sector7_source'](image)
    audit = pd.DataFrame([{'image_path': str(image), 'image_relpath': image.name, **current}])
    if change == 'bytes':
        image.write_bytes(b'two')
    elif change == 'disappear':
        image.unlink()
    elif change == 'appear':
        image.write_bytes(b'one')
    else:
        monkeypatch.setitem(ns, '_p3_file_hash', lambda path: (_ for _ in ()).throw(OSError('unavailable')))
    with pytest.raises(RuntimeError, match='changed during'):
        ns['io_validate_sector7_source_audit'](audit)


@pytest.mark.parametrize('corrupt', ['state_json', 'state_shape', 'parquet', 'missing_schema', 'missing_state'])
def test_corrupt_local_cache_rebuild_preserves_replaced_files_and_history(ns, tmp_path, corrupt):
    _, _, config, tables, lineage = sample(ns, tmp_path)
    saved = ns['io_persist_sector7'](tables, config, lineage)
    root = Path(saved['root'])
    signature = ns['sector7_processing_signature'](config, lineage)
    assert ns['io_load_cached_sector7'](config, signature) is not None
    historical = root / saved['state']['config_snapshot']
    history_bytes = historical.read_bytes()
    if corrupt == 'missing_state':
        broken = root / ns['S7_STATE_PATH']
        broken.unlink()
    elif corrupt in {'state_json', 'state_shape'}:
        broken = root / ns['S7_STATE_PATH']
        broken.write_text('{broken' if corrupt == 'state_json' else '[]')
    elif corrupt == 'parquet':
        broken = root / saved['state']['tables']['image_qc_manifest']['relative_path']
        broken.write_bytes(b'bad-parquet')
    else:
        broken = root / saved['state']['tables']['image_qc_manifest']['schema_path']
        broken.unlink()
    broken_bytes = broken.read_bytes() if broken.exists() else None
    assert ns['io_load_cached_sector7'](config, signature) is None
    repaired = ns['io_persist_sector7'](tables, config, lineage)
    backup = Path(repaired['cache_recovery']['backup_root'])
    assert backup.is_dir()
    if broken_bytes is not None:
        assert (backup / broken.relative_to(root)).read_bytes() == broken_bytes
    assert historical.read_bytes() == history_bytes
    assert ns['_p7_read_state'](root) == repaired['state']
    assert ns['io_load_cached_sector7'](config, signature) is not None
    assert all('recovery' not in r['relative_path'] for r in repaired['state']['artifacts'])
    assert ns['io_persist_sector7'](tables, config, lineage)['status'] == 'unchanged'


@pytest.mark.parametrize('changed', ['upstream', 'source'])
def test_cache_recovery_cannot_suppress_source_or_upstream_changes(ns, tmp_path, changed):
    image, upstream, config, tables, lineage = sample(ns, tmp_path)
    saved = ns['io_persist_sector7'](tables, config, lineage)
    state_path = Path(saved['root']) / ns['S7_STATE_PATH']
    state_path.write_text('{corrupt')
    (upstream if changed == 'upstream' else image).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='changed'):
        ns['io_persist_sector7'](tables, config, lineage)
    assert state_path.read_text() == '{corrupt'
    assert not list(Path(saved['root']).parent.glob('.sector7-recovery-*'))


def test_access_errors_are_not_treated_as_cache_corruption(ns, tmp_path, monkeypatch):
    monkeypatch.setitem(ns, '_p7_read_state', lambda root: (_ for _ in ()).throw(PermissionError('denied')))
    with pytest.raises(PermissionError):
        ns['_p7_local_cache_snapshot'](tmp_path)


def test_failed_recovery_install_restores_original_files(ns, tmp_path, monkeypatch):
    _, _, config, tables, lineage = sample(ns, tmp_path)
    saved = ns['io_persist_sector7'](tables, config, lineage)
    root = Path(saved['root'])
    state_path = root / ns['S7_STATE_PATH']
    state_path.write_text('{corrupt')
    originals = {path.relative_to(root): path.read_bytes()
                 for path in root.rglob('*') if path.is_file()}
    monkeypatch.setitem(ns, '_p3_read_table',
                        lambda *args: (_ for _ in ()).throw(RuntimeError('simulated readback failure')))
    with pytest.raises(RuntimeError, match='simulated readback failure'):
        ns['io_persist_sector7'](tables, config, lineage)
    restored = {path.relative_to(root): path.read_bytes()
                for path in root.rglob('*') if path.is_file()}
    assert restored == originals
    assert not list(root.parent.glob('.sector7-recovery-*'))
