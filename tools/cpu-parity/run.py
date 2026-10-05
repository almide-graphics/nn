#!/usr/bin/env python3
"""Fail-closed numerical prefill gate, against immutable sequential nn (not llama).
Requires NumPy, official gguf-py, Cargo. No network or model download is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import platform
import uuid
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE_REV = '3a75481915c6faa4c831d72d4a0fec433a08a852'
BASE_SHA = '7501f5e7d8eb98691c27678c01a8257fc3a4d8f3ef49033ceb93319b0b15589d'
REAL_SHA = '3d1cfc4b845efcda96d63415c986d51781d02b2f55f93fc6add142cd56b4e314'
ABS_TOL = 1e-3
NORMALIZED_TOL = 1e-4


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def validate_threads(threads):
    available = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    if not threads or len(set(threads)) != len(threads) or any(t < 1 or t > available for t in threads):
        raise ValueError(f'threads must be distinct and between 1 and affinity limit {available}; use --threads 1 on small runners')
    # The Rust driver additionally checks available_parallelism (including cgroup limits)
    # before reading model bytes or allocating the oracle's per-thread scratch arrays.


def forced(seed, n, vocab):
    # Fixed integer arithmetic, independent of logits, language, NumPy PRNG, and engine.
    out = []
    for _ in range(n):
        seed = (seed*6364136223846793005 + 1442695040888963407) & ((1 << 64)-1)
        out.append((seed >> 32) % vocab)
    return out


def make_cases(vocab, tiny):
    fixture = json.loads((HERE/'prompts.json').read_text())
    cases = []
    def add(name, tokens, ends=None):
        tokens = [t % vocab if tiny else t for t in tokens]
        assert all(0 <= t < vocab for t in tokens)
        cases.append(dict(name=name, tokens=tokens,
            continuation=forced(0x504152495459+len(cases), min(16,2048-len(tokens)), vocab),
            chunk_ends=ends or [len(tokens)]))
    for name,tokens in fixture['prompts'].items(): add(name,tokens)
    sequence = forced(42,2048,vocab)
    for n in [1,3,4,5,63,64,65,127,128,129,511,512,513,1024,2032,2048]:
        add(f'length-{n}',sequence[:n])
    for n in [128,1024]: add(f'input-{n}',fixture[f'input_{n}'])
    for name,n,ends in [('split-65',65,[1,65]),('split-129',129,[63,64,129]),
                        ('split-513',513,[128,513])]:
        add(name,sequence[:n],ends)
    return cases


def metadata(model):
    import gguf
    r = gguf.GGUFReader(str(model))
    def val(k): return r.fields['qwen3.'+k].contents()
    tensors = {t.name:t for t in r.tensors}
    assert r.fields['general.architecture'].contents() == 'qwen3'
    assert 'output.weight' not in tensors, 'native executor assumes tied embeddings'
    def offset(name,typ):
        t = tensors[name]
        assert int(t.tensor_type)==typ, (name,t.tensor_type)
        return int(t.data_offset)
    nl = int(val('block_count'))
    cfg = [nl,int(val('attention.head_count')),int(val('attention.head_count_kv')),
           int(val('attention.key_length')),int(val('feed_forward_length')),
           int(val('embedding_length')),int(tensors['token_embd.weight'].shape[1]),
           float(val('rope.freq_base')),float(val('attention.layer_norm_rms_epsilon')),
           offset('token_embd.weight',8),offset('output_norm.weight',0)]
    gs = []; ws = []
    for l in range(nl):
        p = f'blk.{l}.'
        gs += [offset(p+k+'.weight',0) for k in ['attn_norm','attn_q_norm','attn_k_norm','ffn_norm']]
        ws += [offset(p+k+'.weight',8) for k in ['attn_q','attn_k','attn_v','attn_output','ffn_gate','ffn_up','ffn_down']]
    return cfg,gs,ws


def build(out, reference):
    snapshots = out/'sources'; snapshots.mkdir(exist_ok=True)
    if reference:
        original = reference.read_bytes()
    else:
        original = subprocess.check_output(['git','show',BASE_REV+':native/cpu_fast.rs'],cwd=REPO)
        original = original.replace(b'&Vec<u8>', b'&[u8]').replace(b'&Vec<i64>', b'&[i64]')
        original = original.replace(b'impl std::ops::Deref<Target = Vec<u8>>',b'&[u8]')
    assert hashlib.sha256(original).hexdigest() == BASE_SHA, 'immutable baseline hash mismatch'
    (snapshots/'original.rs').write_bytes(original)
    shutil.copyfile(REPO/'native/cpu_fast.rs', snapshots/'candidate.rs')
    env = os.environ.copy()
    env.update(CPU_PARITY_ORIGINAL=str(snapshots/'original.rs'),
               CPU_PARITY_CANDIDATE=str(snapshots/'candidate.rs'), CARGO_TARGET_DIR=str(out/'target'))
    # Same optimized code generation as the original Almide --fast harness.
    env['RUSTFLAGS'] = '-C target-cpu=native'
    with (out/'build.log').open('w') as log:
        subprocess.run(['cargo','build','--release','--offline','--manifest-path',str(HERE/'Cargo.toml')],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary = out/'target/release/cpu-prefill-parity'
    return dict(reference_revision=BASE_REV, reference_sha256=sha(snapshots/'original.rs'),
        candidate_sha256=sha(snapshots/'candidate.rs'), binary_sha256=sha(binary),
        driver_sha256=sha(HERE/'src/main.rs'), runner_sha256=sha(HERE/'run.py'),
        rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
        rustflags=env['RUSTFLAGS'], binary=str(binary))


def compare(out, manifest):
    import numpy as np
    records = []; missing = []; v = manifest['vocab']
    for name in ['config.txt','cases.tsv']:
        if sha(out/name) != manifest.get('input_sha256',{}).get(name):
            missing.append(name+': input fingerprint missing or changed')
    for nt in manifest['threads']:
        base = out/f't{nt}'
        for mode in ['reference','candidate']:
            if not (base/mode/'complete').exists(): missing.append(str(base/mode/'complete'))
            stamp = base/mode/'execution.json'
            if not stamp.exists():
                missing.append(str(stamp))
            else:
                execution = json.loads(stamp.read_text())
                if (not manifest.get('run_id') or execution.get('run_id') != manifest['run_id']
                    or execution.get('binary_sha256') != manifest['build']['binary_sha256']
                    or execution.get('threads') != nt or execution.get('mode') != mode):
                    missing.append(str(stamp)+': stale or wrong execution')
        for case in manifest['cases']:
            name = case['name']; checkpoints = len(case['chunk_ends'])+len(case['continuation'])
            paths = [base/mode/f'{name}.f32' for mode in ['reference','candidate']]
            topfile = base/'candidate'/f'{name}.argmax'
            for p in paths+[topfile]:
                if not p.exists(): missing.append(str(p))
            if any(not p.exists() for p in paths+[topfile]): continue
            try:
                ref,cur = [np.fromfile(p,dtype='<f4') for p in paths]
                tops = [int(t) for t in topfile.read_text().split()]
            except (OSError,ValueError) as e:
                missing.append(f'{name}: unreadable dump: {e}');continue
            if len(ref)!=checkpoints*v or cur.shape!=ref.shape or len(tops)!=checkpoints:
                missing.append(f'{name}: shape mismatch'); continue
            ref=ref.reshape(checkpoints,v);cur=cur.reshape(checkpoints,v)
            for cp,(r,c) in enumerate(zip(ref,cur)):
                finite=bool(np.isfinite(r).all() and np.isfinite(c).all())
                err=float(np.max(np.abs(r.astype('float64')-c.astype('float64')))) if finite else None
                scale=float(np.max(np.abs(r))) if finite else None
                norm=err/max(scale,float(np.finfo('float32').tiny)) if finite else None
                rt=int(np.argmax(r)); ct=int(np.argmax(c))
                passed=finite and err<=ABS_TOL and norm<=NORMALIZED_TOL and rt==ct==tops[cp]
                records.append(dict(case=name,threads=nt,checkpoint=cp,
                    position=(case['chunk_ends'][cp]-1 if cp<len(case['chunk_ends']) else
                              len(case['tokens'])+cp-len(case['chunk_ends'])),
                    finite=finite,max_abs_error=err,normalized_max_error=norm,
                    bit_identical=bool(np.array_equal(r.view('u4'),c.view('u4'))),
                    reference_top1=rt,candidate_logits_top1=ct,candidate_argmax_top1=tops[cp],passed=bool(passed)))
    failed=[r for r in records if not r['passed']]
    report=dict(passed=not missing and not failed and bool(records), full_suite=manifest['full_suite'],
        tested_model=manifest['model'],model_sha256=manifest['model_sha256'],
        candidate_sha256=manifest['build']['candidate_sha256'],
        reference_sha256=BASE_SHA,thresholds=manifest['thresholds'],
        cases=len(manifest['cases']),threads=manifest['threads'],checkpoints=len(records),
        all_bit_identical=bool(records) and all(r['bit_identical'] for r in records),
        max_abs_error=max((r['max_abs_error'] for r in records if r['finite']),default=None),
        max_normalized_error=max((r['normalized_max_error'] for r in records if r['finite']),default=None),
        top1_matches=sum(r['reference_top1']==r['candidate_logits_top1']==r['candidate_argmax_top1'] for r in records),
        failed_checkpoints=len(failed),missing=missing,failures=failed,results=records)
    save(out/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','failures')},indent=2))
    return report['passed']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=REPO/'testdata/tiny-qwen3-q8_0.gguf')
    p.add_argument('--gguf-py',type=Path,help='path to official llama.cpp/gguf-py')
    p.add_argument('--reference',type=Path,help='optional immutable signature-adapted original source')
    p.add_argument('--real-model',action='store_true',help='explicit opt-in for non-tiny model')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',default='1,4')
    p.add_argument('--case-regex',default='.*',help='partial runs are marked full_suite=false')
    p.add_argument('--phase',choices=['all','prepare','reference','candidate','compare'],default='all')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.phase in ['all','prepare']:
        selected_threads=[int(t) for t in args.threads.split(',')]
        validate_threads(selected_threads)
        if args.gguf_py:sys.path.insert(0,str(args.gguf_py.resolve()))
        model=args.model.resolve(); tiny=sha(model)==sha(REPO/'testdata/tiny-qwen3-q8_0.gguf')
        assert tiny or args.real_model, 'non-tiny weights require --real-model'
        if not tiny: assert sha(model)==REAL_SHA, 'unexpected real model SHA256'
        cfg,gs,ws=metadata(model)
        cases=[c for c in make_cases(cfg[6],tiny) if re.fullmatch(args.case_regex,c['name'])]
        assert cases,'empty selection'
        for marker in out.glob('t*/*/complete'): marker.unlink()
        for marker in out.glob('t*/*/execution.json'): marker.unlink()
        manifest=dict(run_id=uuid.uuid4().hex,platform=platform.platform(),
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            model=str(model),model_sha256=sha(model),vocab=cfg[6],tiny=tiny,
            threads=selected_threads,cases=cases,
            full_suite=args.case_regex=='.*',thresholds=dict(max_absolute=ABS_TOL,
                max_error_over_max_abs_reference=NORMALIZED_TOL,exact_top1=True,finite_only=True),
            comparison='candidate batched prefill and continuation vs immutable sequential nn',
            continuation_policy='16 teacher-forced LCG IDs, or remaining positions at 2048 limit',
            build=build(out,args.reference))
        (out/'config.txt').write_text('\n'.join(' '.join(map(str,x)) for x in [cfg,gs,ws])+'\n')
        (out/'cases.tsv').write_text(''.join(c['name']+'\t'+'\t'.join(' '.join(map(str,c[k]))
                                    for k in ['tokens','continuation','chunk_ends'])+'\n' for c in cases))
        manifest['input_sha256']={name:sha(out/name) for name in ['config.txt','cases.tsv']}
        save(out/'manifest.json',manifest)
        if args.phase=='prepare':return
    else:manifest=json.loads((out/'manifest.json').read_text())
    if args.phase in ['all','reference','candidate']:
        validate_threads(manifest['threads'])
        assert sha(Path(manifest['model']))==manifest['model_sha256'],'model changed'
        assert sha(Path(manifest['build']['binary']))==manifest['build']['binary_sha256'],'binary changed'
        for nt in manifest['threads']:
            for mode in (['reference','candidate'] if args.phase=='all' else [args.phase]):
                dest=out/f't{nt}'/mode;dest.mkdir(parents=True,exist_ok=True)
                (dest/'complete').unlink(missing_ok=True)
                (dest/'execution.json').unlink(missing_ok=True)
                env=os.environ.copy();env.update(ALMIDE_LOCKSTEP_THREADS=str(nt),RAYON_NUM_THREADS=str(nt))
                with (dest/'stderr.log').open('w') as log:
                    subprocess.run([manifest['build']['binary'],mode,manifest['model'],str(out/'config.txt'),
                                    str(out/'cases.tsv'),str(dest)],env=env,stderr=log,check=True)
                save(dest/'execution.json',dict(run_id=manifest['run_id'],mode=mode,threads=nt,
                    binary_sha256=manifest['build']['binary_sha256']))
    if args.phase in ['all','compare']:
        sys.exit(0 if compare(out,manifest) else 1)

if __name__=='__main__':main()
