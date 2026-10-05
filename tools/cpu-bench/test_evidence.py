#!/usr/bin/env python3
import copy
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from evidence import ORACLE_SHA, OVERRIDES, sha, validate_quality_summary, validate_build, validate_nn_result

class EvidenceTests(unittest.TestCase):
    def gate(self):
        return dict(passed=True,model_sha256='model',candidate_sha256='source',reference_sha256=ORACLE_SHA,
            thresholds=dict(max_absolute=.001,max_error_over_max_abs_reference=.0001,exact_top1=True,finite_only=True),
            runtime_overrides=dict.fromkeys(OVERRIDES,None),threads=[1],full_suite=False,cases=1,
            failed_checkpoints=0,missing=[],failures=[],checkpoints=1,top1_matches=1,max_abs_error=0.,max_normalized_error=0.,
            all_bit_identical=True,results=[dict(case='test',threads=1,checkpoint=0,passed=True,finite=True,
                max_abs_error=0.,normalized_max_error=0.,reference_top1=0,candidate_logits_top1=0,candidate_argmax_top1=0,bit_identical=True)])
    def validate(self,g):return validate_quality_summary(g,'model','source',1,False)
    def test_valid_summary(self):self.validate(self.gate())
    def test_missing_or_forged_thresholds(self):
        for change in [None,dict(max_absolute=.1,max_error_over_max_abs_reference=.1,exact_top1=True,finite_only=True)]:
            g=self.gate();g['thresholds']=change
            with self.assertRaises(ValueError):self.validate(g)
    def test_wrong_oracle_source_model(self):
        for key in ['reference_sha256','candidate_sha256','model_sha256']:
            g=self.gate();g[key]='wrong'
            with self.assertRaises(ValueError):self.validate(g)
    def test_hidden_failed_point(self):
        for change in [dict(passed=False),dict(finite=False),dict(max_abs_error=.002),dict(normalized_max_error=.0002),
                       dict(candidate_argmax_top1=1),dict(max_abs_error=float('nan'))]:
            g=self.gate();g['results'][0].update(change)
            with self.assertRaises(ValueError):self.validate(g)
    def test_override_or_missing_path_evidence(self):
        for value in [{},dict.fromkeys(OVERRIDES,'avx2')]:
            g=self.gate();g['runtime_overrides']=value
            with self.assertRaises(ValueError):self.validate(g)
    def test_false_full_suite_claim(self):
        g=self.gate();g['full_suite']=True
        with self.assertRaises(ValueError):validate_quality_summary(g,'model','source',1,True)
    def test_mode(self):
        m=dict(engine='nn',prefill_mode='batched',n_prompt=128,n_decode=128,threads=1,ids=[0]*129)
        validate_nn_result(m,128,1)
        for change in [dict(prefill_mode='sequential'),dict(prefill_mode=None),dict(n_decode=127),dict(ids=[0]*128)]:
            with self.assertRaises(ValueError):validate_nn_result({**m,**change},128,1)
    def test_build_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binary=root/'nn';binary.write_bytes(b'binary')
            sidecar=root/'nn.build.json';snapshot=root/'nn.build-files';snapshot.mkdir()
            inputs={}
            for name in ['native/cpu_fast.rs','examples/_duel_prefill.almd','almide.toml']:
                f=snapshot/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(name);inputs[name]=sha(f)
            data=dict(schema='nn-cpu-build-v1',status='complete',binary_sha256=sha(binary),
                compiler=dict(version='almide 0.66.0 (release, 819bbc74f)',path='/compiler',sha256='0'*64),
                rustc=dict(version='rustc 1.94.0 (test)',path='/rustc',sha256='1'*64),
                cargo=dict(version='cargo 1.94.0 (test)',path='/cargo',sha256='2'*64),
                snapshot_directory=snapshot.name,inputs=inputs,candidate_source_sha256=inputs['native/cpu_fast.rs'],
                harness_sha256=inputs['examples/_duel_prefill.almd'],config_sha256=inputs['almide.toml'],
                command=['/compiler','build','examples/_duel_prefill.almd','--fast','-o',str(binary)])
            sidecar.write_text(json.dumps(data));validate_build(sidecar,binary)
            binary.write_bytes(b'wrong binary')
            with self.assertRaises(ValueError):validate_build(sidecar,binary)
            binary.write_bytes(b'binary');(snapshot/'native/cpu_fast.rs').write_text('wrong source')
            with self.assertRaises(ValueError):validate_build(sidecar,binary)
    def test_builder_cache_and_generated_lock_metadata(self):
        spec=importlib.util.spec_from_file_location('builder',Path(__file__).with_name('build-nn-harness.py'))
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        self.assertEqual(builder.record_environment({'CARGO_TARGET_DIR':'/cache'})['CARGO_TARGET_DIR'],'/cache')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);tmp=root/'tmp';snapshot=root/'snapshot';(tmp/'almide-run').mkdir(parents=True)
            for name in ['Cargo.toml','Cargo.lock']:(tmp/'almide-run'/name).write_text(name)
            records=builder.collect_generated_cargo(tmp,snapshot)
            self.assertEqual({Path(r['snapshot_path']).name for r in records},{'Cargo.toml','Cargo.lock'})
            for row in records:self.assertEqual(sha(snapshot/row['snapshot_path']),row['sha256'])
    def test_missing_sidecar(self):
        with self.assertRaises(OSError):validate_build('/definitely/missing/build-sidecar','/missing/binary')

if __name__=='__main__':unittest.main()
