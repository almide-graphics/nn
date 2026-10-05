#!/usr/bin/env python3
"""Cheap regression tests that the numerical gate really fails closed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('gate',Path(__file__).with_name('run.py'))
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

class GateTests(unittest.TestCase):
    def run_gate(self,ref,cur,top=0,stale=False,truncated=False,finite=True):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['config.txt','cases.tsv']:(root/name).write_text('fixture\n')
            manifest=dict(run_id='test',model='synthetic',model_sha256='test',vocab=2,
                build=dict(candidate_sha256='candidate',binary_sha256='binary'),
                thresholds=dict(max_absolute=gate.ABS_TOL,max_error_over_max_abs_reference=gate.NORMALIZED_TOL),
                full_suite=False,threads=[1],cases=[dict(name='test',tokens=[0],continuation=[],chunk_ends=[1])],
                input_sha256={name:gate.sha(root/name) for name in ['config.txt','cases.tsv']})
            for mode,values in [('reference',ref),('candidate',cur)]:
                p=root/'t1'/mode;p.mkdir(parents=True)
                np.array(values,dtype='<f4').tofile(p/'test.f32')
                (p/'complete').write_text('ok\n')
                gate.save(p/'execution.json',dict(run_id='stale' if stale else 'test',mode=mode,threads=1,binary_sha256='binary'))
            (root/'t1/candidate/test.argmax').write_text(str(top)+'\n')
            if truncated:(root/'t1/candidate/test.f32').write_bytes(b'')
            with contextlib.redirect_stdout(io.StringIO()):passed=gate.compare(root,manifest)
            return passed,json.loads((root/'summary.json').read_text())

    def sample_gate(self,ids=(0,0),cur=(2.,1.),missing=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            manifest=dict(threads=[1],vocab=2,sample_cases=[dict(name='sample',chunk_ends=[1],continuation=[0],temp=.6,top_p=.9,seed=42)])
            for mode,values,sampled in [('reference',[2.,1.],[0,0]),('candidate',cur,ids)]:
                p=root/'t1'/mode;p.mkdir(parents=True)
                (p/'sampled-complete').write_text('ok\n')
                (p/'sample.sampled.ids').write_text(' '.join(map(str,sampled)))
                np.array(values,dtype='<f4').tofile(p/'sample.sampled.f32')
            if missing:(root/'t1/candidate/sample.sampled.ids').unlink()
            return gate.compare_samples(root,manifest)['passed']
    def test_sampled_exact_passes(self):self.assertTrue(self.sample_gate())
    def test_sampled_wrong_ids_fail(self):self.assertFalse(self.sample_gate(ids=(0,1)))
    def test_sampled_error_fails(self):self.assertFalse(self.sample_gate(cur=(2.002,1.)))
    def test_sampled_missing_fails(self):self.assertFalse(self.sample_gate(missing=True))
    def test_thread_guards(self):
        gate.validate_threads([1])
        for threads in [[],[0],[1,1],[10**9]]:
            with self.assertRaises(ValueError):gate.validate_threads(threads)
    def test_exact_passes(self):
        passed,report=self.run_gate([2,1],[2,1]);self.assertTrue(passed);self.assertTrue(report['all_bit_identical'])
    def test_zero_reference(self):
        passed,_=self.run_gate([0,0],[0,0]);self.assertTrue(passed)
    def test_absolute_error_fails_independently(self):
        passed,_=self.run_gate([100,0],[100.002,0]);self.assertFalse(passed)
    def test_normalized_error_fails_independently(self):
        passed,_=self.run_gate([1,0],[1.0002,0]);self.assertFalse(passed)
    def test_logit_top1_fails_within_numeric_tolerances(self):
        passed,_=self.run_gate([1,1],[1,1.000001],top=1);self.assertFalse(passed)
    def test_production_argmax_fails(self):
        passed,_=self.run_gate([2,1],[2,1],top=1);self.assertFalse(passed)
    def test_nan_fails(self):
        passed,_=self.run_gate([2,1],[float('nan'),1]);self.assertFalse(passed)
    def test_inf_fails(self):
        passed,_=self.run_gate([2,1],[float('inf'),1]);self.assertFalse(passed)
    def test_stale_run_fails(self):
        passed,_=self.run_gate([2,1],[2,1],stale=True);self.assertFalse(passed)
    def test_truncated_dump_fails(self):
        passed,_=self.run_gate([2,1],[2,1],truncated=True);self.assertFalse(passed)

if __name__=='__main__':unittest.main()
