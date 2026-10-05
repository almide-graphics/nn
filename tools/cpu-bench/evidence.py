"""Read-only, fail-closed evidence checks. Never execute commands from a sidecar."""
import hashlib
import json
import math
import re
from pathlib import Path

ORACLE_SHA='7501f5e7d8eb98691c27678c01a8257fc3a4d8f3ef49033ceb93319b0b15589d'
ABS_TOL=1e-3
NORM_TOL=1e-4
OVERRIDES=['NN_PREFILL_PROFILE','NN_PREFILL_KERNEL','NN_PREFILL_ATTENTION']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def require(condition,message):
    if not condition:raise ValueError(message)

def finite_bound(value,limit):
    return type(value) in (int,float) and math.isfinite(value) and 0<=value<=limit

def validate_quality_summary(gate,model_sha,source_sha,threads,confirmatory):
    require(gate.get('passed') is True,'numerical gate did not pass')
    require(gate.get('model_sha256')==model_sha and gate.get('candidate_sha256')==source_sha,'gate source/model mismatch')
    require(gate.get('reference_sha256')==ORACLE_SHA,'wrong or missing immutable oracle')
    require(gate.get('thresholds')==dict(max_absolute=ABS_TOL,max_error_over_max_abs_reference=NORM_TOL,
        exact_top1=True,finite_only=True),'wrong or missing fixed thresholds')
    require(gate.get('runtime_overrides')==dict.fromkeys(OVERRIDES,None),'gate did not verify default runtime path')
    require(threads in gate.get('threads',[]),'gate omitted requested thread count')
    require(not confirmatory or gate.get('full_suite') is True,'confirmatory run requires full suite')
    require(gate.get('failed_checkpoints')==0 and gate.get('missing')==[] and gate.get('failures')==[],
        'failed, missing, or unreported gate checkpoints')
    rows=gate.get('results');n=gate.get('checkpoints')
    require(type(n) is int and n>0 and isinstance(rows,list) and len(rows)==n,'invalid checkpoint count')
    require(gate.get('top1_matches')==n,'top1 mismatch')
    require(finite_bound(gate.get('max_abs_error'),ABS_TOL) and finite_bound(gate.get('max_normalized_error'),NORM_TOL),
        'invalid aggregate error')
    if confirmatory:
        names={f'prompt-{i:02}' for i in range(20)} | {f'length-{i}' for i in [1,3,4,5,63,64,65,127,128,129,511,512,513,1024,2032,2048]} | {'input-128','input-1024','split-65','split-129','split-513'}
        require(gate.get('cases')==41 and n==685*len(gate['threads']) and {r.get('case') for r in rows}==names,
            'full-suite claim lacks fixed boundary/prompt coverage')
    seen=set()
    for row in rows:
        key=(row.get('threads'),row.get('case'),row.get('checkpoint'))
        require(key not in seen,'duplicate checkpoint');seen.add(key)
        require(row.get('threads') in gate['threads'],'unexpected thread count in results')
        require(row.get('passed') is True and row.get('finite') is True,'failed or non-finite checkpoint')
        require(finite_bound(row.get('max_abs_error'),ABS_TOL) and finite_bound(row.get('normalized_max_error'),NORM_TOL),
            'checkpoint exceeds fixed tolerances')
        tops=[row.get(k) for k in ['reference_top1','candidate_logits_top1','candidate_argmax_top1']]
        require(all(type(t) is int and t>=0 for t in tops) and tops[0]==tops[1]==tops[2],'checkpoint top1 mismatch')
    require(gate['max_abs_error']==max(r['max_abs_error'] for r in rows) and
        gate['max_normalized_error']==max(r['normalized_max_error'] for r in rows),'aggregate errors disagree with results')
    if gate.get('all_bit_identical') is True:
        require(all(r.get('bit_identical') is True for r in rows),'bit-identity claim disagrees with results')
    if 'sampling' in gate:require(gate['sampling'].get('passed') is True,'sampled gate failed')
    return seen

def validate_quality(path,model_sha,source_sha,threads,confirmatory):
    path=Path(path);gate=json.loads(path.read_text())
    seen=validate_quality_summary(gate,model_sha,source_sha,threads,confirmatory)
    root=path.parent;manifest=json.loads((root/'manifest.json').read_text());build=manifest['build']
    require(gate.get('manifest_sha256')==sha(root/'manifest.json'),'gate manifest missing or changed')
    require(gate.get('run_id')==manifest.get('run_id') and bool(manifest.get('run_id')),'gate run identity mismatch')
    require(manifest.get('runtime_overrides')==gate['runtime_overrides'],'runtime path metadata mismatch')
    require(manifest.get('model_sha256')==model_sha and build.get('candidate_sha256')==source_sha and
        build.get('reference_sha256')==ORACLE_SHA,'gate manifest source/model mismatch')
    require(sha(root/'sources/original.rs')==ORACLE_SHA and sha(root/'sources/candidate.rs')==source_sha,
        'gate source snapshot changed')
    require(sha(Path(build['binary']))==build['binary_sha256'],'gate binary changed')
    require(manifest.get('threads')==gate['threads'] and manifest.get('full_suite')==gate['full_suite'],
        'gate coverage metadata mismatch')
    require(len(manifest['cases'])==gate['cases'],'case count mismatch')
    expected={(nt,c['name'],cp) for nt in manifest['threads'] for c in manifest['cases']
              for cp in range(len(c['chunk_ends'])+len(c['continuation']))}
    require(seen==expected,'incomplete checkpoint coverage')
    for name,digest in manifest['input_sha256'].items():
        require(Path(name).name==name and sha(root/name)==digest,'gate input changed')
    require({'config.txt','cases.tsv'}<=set(manifest['input_sha256']),'gate input fingerprints missing')
    for nt in manifest['threads']:
        for mode in ['reference','candidate']:
            p=root/f't{nt}'/mode;stamp=json.loads((p/'execution.json').read_text())
            require((p/'complete').read_text().strip()=='ok','gate run did not complete')
            require(stamp==dict(run_id=manifest['run_id'],mode=mode,threads=nt,binary_sha256=build['binary_sha256']),
                'stale execution stamp')
    return gate

def validate_build(sidecar,binary):
    sidecar=Path(sidecar);data=json.loads(sidecar.read_text())
    require(data.get('schema')=='nn-cpu-build-v1' and data.get('status')=='complete','missing successful build attestation')
    require(data.get('binary_sha256')==sha(binary),'benchmark binary differs from build attestation')
    for name in ['compiler','rustc','cargo']:
        info=data.get(name,{})
        require(bool(re.fullmatch('[0-9a-f]{64}',info.get('sha256',''))) and Path(info.get('path','')).is_absolute(),
            f'missing {name} identity/hash')
    require(data.get('cargo',{}).get('version','').startswith('cargo 1.94.0 '),'wrong Cargo version')
    require(data.get('compiler',{}).get('version')=='almide 0.66.0 (release, 819bbc74f)','wrong Almide compiler')
    require(data.get('rustc',{}).get('version','').startswith('rustc 1.94.0 '),'wrong Rust compiler')
    require(Path(data['snapshot_directory']).name==data['snapshot_directory'],'snapshot must be adjacent to sidecar')
    root=(sidecar.parent/data['snapshot_directory']).resolve()
    inputs=data.get('inputs',{})
    for required in ['native/cpu_fast.rs','examples/_duel_prefill.almd','almide.toml']:
        require(required in inputs,'required build input missing')
    for relative,digest in inputs.items():
        p=(root/relative).resolve();require(p.is_relative_to(root),'invalid snapshot path')
        require(sha(p)==digest,f'build snapshot changed: {relative}')
    require(data.get('candidate_source_sha256')==inputs['native/cpu_fast.rs'],'source attestation mismatch')
    require(data.get('harness_sha256')==inputs['examples/_duel_prefill.almd'],'harness attestation mismatch')
    require(data.get('config_sha256')==inputs['almide.toml'],'config attestation mismatch')
    cmd=data.get('command',[])
    require(len(cmd)==6 and cmd[1:4]==['build','examples/_duel_prefill.almd','--fast'] and cmd[4]=='-o',
        'unexpected attested build command')
    for record in data.get('generated_cargo',[]):
        path=(root/record['snapshot_path']).resolve()
        require(path.is_relative_to(root) and sha(path)==record['sha256'],'generated Cargo config changed')
    # Intentionally never run cmd, compiler paths, or any other sidecar content.
    return data

def validate_nn_result(result,n,threads):
    require(result.get('engine')=='nn' and result.get('prefill_mode')=='batched','wrong nn engine/prefill mode')
    require(result.get('n_prompt')==n and result.get('n_decode')==128 and result.get('threads')==threads,'wrong nn work count')
    require(len(result.get('ids',[]))==129 and all(type(t) is int and t>=0 for t in result['ids']),'wrong emitted nn IDs')
