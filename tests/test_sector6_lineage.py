"""Sector 6 provenance regression without GPU, Google Drive, or notebook execution.

Run: python -m unittest discover -s tests -p 'test_sector6_lineage.py'
Requires the notebook's CPU dependencies: numpy, pandas, opencv, Pillow, tqdm.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import pandas as pd

NOTEBOOK = (Path(__file__).resolve().parents[1] / 'notebooks' / 'phase2' /
            'phase2_06_temporal_context_images.ipynb')


def load_notebook(path):
    """Load declarations only; never install packages, mount Drive, or run pipeline."""
    namespace = {'__name__': 'sector6_notebook_tests'}
    for cell in json.loads(path.read_text())['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] != 'code' or source.lstrip().startswith('%'):
            continue
        tree = ast.parse(source)
        declares_functions = any(isinstance(node, ast.FunctionDef) for node in tree.body)
        declarations = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import)):
                declarations.append(node)
            elif isinstance(node, ast.ImportFrom) and node.module != 'IPython.display':
                declarations.append(node)
            elif (declares_functions and isinstance(node, ast.Assign)
                  and all(isinstance(target, ast.Name) for target in node.targets)):
                # Definitions use literals, previous constants, and frozenset;
                # excluding other calls prevents accidental execution of I/O.
                calls = [item for item in ast.walk(node.value) if isinstance(item, ast.Call)]
                if all(isinstance(call.func, ast.Name) and call.func.id == 'frozenset'
                       for call in calls):
                    declarations.append(node)
        exec(compile(ast.Module(body=declarations, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def definitions(path):
    found = set()
    for cell in json.loads(path.read_text())['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and not source.lstrip().startswith('%'):
            found.update(node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    return found


class Sector6LineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_notebook(NOTEBOOK)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.official = self.root/'images/target.jpg'
        self.official.parent.mkdir()
        cv2.imwrite(str(self.official), np.arange(24*32*3, dtype=np.uint8).reshape(24,32,3))
        self.video = self.root/'videos/video-A.mp4'
        self.video.parent.mkdir()
        # NVDEC is stubbed, but source hashes use real bytes.
        self.video.write_bytes(b'fixture-video-content')
        self.labelled = pd.DataFrame([dict(
            annotation_id='ann-0', video_key='video-A', frame_number=10,
            opencv_frame_index=10, frame_index_offset=0,
            finding_class_normalized='bleeding', image_path=str(self.official),
            image_relpath='images/target.jpg')])
        self.videos = pd.DataFrame([dict(video_key='video-A', frame_count=100,
            container_fps=2.0, video_path=str(self.video), video_relpath='videos/video-A.mp4')])
        self.segments = pd.DataFrame([dict(
            finding_segment_id='segment-A', video_key='video-A', finding_class='bleeding',
            finding_class_normalized='bleeding', split='train', target_frame_number=10,
            target_opencv_frame_index=10, target_image_annotation_id='ann-0',
            target_image_path=str(self.official), target_image_relpath='images/target.jpg',
            temporal_grounding_candidate=True, temporal_supervision_level='sparse_annotations',
            verified_temporal_boundary_available=False)])
        self.inputs = {'sector3': {'video_frame_manifest':self.videos},
                       'sector5': {'labelled_frame_manifest_with_segments':self.labelled,
                                   'finding_segment_manifest':self.segments}}
        self.config = dict(self.ns['SECTOR6_DEFAULTS'], storage_root=str(self.root),
                           temporal_context_offsets_seconds=[0.0,1.0],
                           image_audit_backend='cpu',image_audit_io_workers=2,
                           image_audit_cpu_workers=2)
        self.lineage = {'input_sha256': {'sector5/manifest':'m'*64},
                        'input_descriptors': {'sector5/manifest':'/dataset/manifest.parquet'},
                        'nested': {'leave_untouched':[1,2,3]}}

    def tearDown(self):
        self.tmp.cleanup()

    def extraction_stub(self, requests, config, decoder_signature):
        rows=[]
        for req in requests.itertuples(index=False):
            path=Path(req.output_path)
            path.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(path),np.full((24,32,3),71,dtype=np.uint8))
            rows.append(dict(video_key=req.video_key,opencv_frame_index=req.opencv_frame_index,
                extracted_image_path=str(path),extracted_image_relpath=req.output_relpath,
                extraction_status='success',extraction_error=pd.NA,
                source_video_sha256=hashlib.sha256(Path(req.video_path).read_bytes()).hexdigest(),
                extracted_image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),decoder_backend='fixture_nvdec'))
        return self.ns['s6_typed_table'](pd.DataFrame(rows,columns=self.ns['TEMPORAL_EXTRACTION_RESULT_COLUMNS']))

    def run_with_real_lineage(self):
        # Only the expensive video decoder is replaced; the lineage function must exist.
        self.assertTrue(callable(self.ns.get('io_s6_extend_lineage')))
        with mock.patch.dict(self.ns,run_temporal_extraction=self.extraction_stub):
            return self.ns['io_run_sector6'](self.inputs,self.config,self.lineage,'fixture-decoder')

    def test_lineage_builder_is_defined_in_the_notebook(self):
        self.assertIn('io_s6_extend_lineage', definitions(NOTEBOOK))
        self.assertTrue(callable(self.ns.get('io_s6_extend_lineage')))

    def test_real_orchestrator_binds_sources_and_excludes_extracted_outputs(self):
        before=copy.deepcopy(self.lineage)
        result=self.run_with_real_lineage()
        self.assertEqual(self.lineage,before)
        lineage=result['lineage']
        self.assertEqual(lineage['input_sha256'],dict(before['input_sha256'],**{
            'source_video/video-A':hashlib.sha256(self.video.read_bytes()).hexdigest(),
            'official_image/images/target.jpg':hashlib.sha256(self.official.read_bytes()).hexdigest()}))
        self.assertEqual(lineage['input_descriptors']['source_video/video-A'],str(self.video))
        self.assertEqual(lineage['input_descriptors']['official_image/images/target.jpg'],str(self.official))
        tables=result['tables']
        self.assertEqual(len(tables['temporal_manifest']),2)
        self.assertEqual(len(tables['temporal_extraction_requests']),1)
        self.assertEqual(tables['temporal_image_audit'].image_readable.tolist(),[True,True])
        self.assertTrue(self.ns['build_sector6_summary'](tables)['validation_passed'])
        extracted=set(tables['extracted_temporal_frames'].extracted_image_relpath)
        self.assertTrue(all('official_image/'+path not in lineage['input_sha256'] for path in extracted))
        lineage['nested']['leave_untouched'].append(4)
        self.assertEqual(self.lineage,before)

    def test_real_orchestrator_preserves_image_audit_diagnostics(self):
        result=self.run_with_real_lineage()
        self.assertEqual(result['lineage']['image_audit_execution'],
                         result['tables']['temporal_image_audit'].attrs['image_audit'])
        self.assertEqual(result['lineage']['image_audit_execution']['cpu_images'],2)

    def test_inconsistent_source_hashes_rejected_without_mutating_input(self):
        before=copy.deepcopy(self.lineage)
        requests=pd.DataFrame([dict(video_key='video-A',video_path=str(self.video))])
        results=pd.DataFrame([dict(video_key='video-A',source_video_sha256='a'*64,
                                  extracted_image_path='/output/a.jpg'),
                              dict(video_key='video-A',source_video_sha256='b'*64,
                                  extracted_image_path='/output/b.jpg')])
        audit=self.ns['s6_empty_table'](self.ns['S6_AUDIT_COLUMNS'])
        with self.assertRaisesRegex(RuntimeError,'inconsistent content hashes'):
            self.ns['io_s6_extend_lineage'](self.lineage,requests,results,audit)
        self.assertEqual(self.lineage,before)

    def test_failed_official_image_is_not_bound_as_valid_input(self):
        requests=pd.DataFrame(columns=['video_key','video_path'])
        results=pd.DataFrame(columns=['video_key','source_video_sha256','extracted_image_path'])
        audit=pd.DataFrame([dict(context_image_path=str(self.official),
            context_image_relpath='images/target.jpg',image_sha256='a'*64,image_readable=False)])
        actual=self.ns['io_s6_extend_lineage'](self.lineage,requests,results,audit)
        self.assertEqual(actual,self.lineage)
        self.assertIsNot(actual,self.lineage)

    def test_missing_lineage_helper_fails_before_expensive_stages(self):
        extraction=mock.Mock(side_effect=AssertionError('Extraction should not start'))
        audit=mock.Mock(side_effect=AssertionError('Audit should not start'))
        with mock.patch.dict(self.ns,run_temporal_extraction=extraction,io_audit_temporal_images=audit):
            helper=self.ns.pop('io_s6_extend_lineage',None)
            try:
                with self.assertRaisesRegex((NameError,RuntimeError),'io_s6_extend_lineage'):
                    self.ns['io_run_sector6'](self.inputs,self.config,self.lineage,'fixture-decoder')
            finally:
                if helper is not None:
                    self.ns['io_s6_extend_lineage']=helper
        extraction.assert_not_called()
        audit.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)
