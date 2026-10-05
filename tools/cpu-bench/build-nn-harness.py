#!/usr/bin/env python3
"""Build a frozen nn harness and write a verifiable source/binary provenance sidecar."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from evidence import sha, require

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PINNED_ALMIDE='almide 0.66.0 (release, 819bbc74f)'

def tool_info(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path),version=subprocess.check_output([str(path),'--version'],text=True).strip())

def record_environment(env):
    return {name:env.get(name) for name in ['PATH','CARGO_HOME','RUSTUP_HOME','CARGO_BUILD_JOBS',
        'CARGO_TARGET_DIR','TMPDIR','RUSTFLAGS']}

def collect_generated_cargo(tmp,snapshot):
    records=[]
    for generated in Path(tmp).glob('*/Cargo.*'):
        target=snapshot/'generated'/generated.parent.name/generated.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(generated,target)
        records.append(dict(original_path=str(generated),snapshot_path=str(target.relative_to(snapshot)),sha256=sha(target)))
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--almide',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    builder_sha=sha(Path(__file__))
    a=p.parse_args();binary=a.out.resolve();sidecar=Path(str(binary)+'.build.json')
    snapshot=Path(str(binary)+'.build-files')
    require(not binary.exists() and not sidecar.exists() and not snapshot.exists(),'use a new output binary path')
    compiler=tool_info(a.almide)
    require(compiler['version']==PINNED_ALMIDE,f'expected {PINNED_ALMIDE}')
    rustc=tool_info(shutil.which('rustc') or 'rustc')
    cargo=tool_info(shutil.which('cargo') or 'cargo')
    require(rustc['version'].startswith('rustc 1.94.0 ') and cargo['version'].startswith('cargo 1.94.0 '),'Rust/Cargo 1.94.0 required')
    files=[REPO/'almide.toml',REPO/'examples/_duel_prefill.almd']
    if (REPO/'almide.lock').exists():files.append(REPO/'almide.lock')
    for dirname in ['src','native']:
        files+=sorted(p for p in (REPO/dirname).rglob('*') if p.is_file())
    inputs={str(f.relative_to(REPO)):sha(f) for f in files}
    snapshot.mkdir(parents=True)
    for relative,digest in inputs.items():
        dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/relative,dest)
        require(sha(dest)==digest and sha(REPO/relative)==digest,'repository changed while freezing inputs; start a new build')
    command=[compiler['path'],'build','examples/_duel_prefill.almd','--fast','-o',str(binary)]
    env=os.environ.copy();env['CARGO_BUILD_JOBS']='2'
    env['RUSTFLAGS']='-C target-cpu=native'
    log=Path(str(binary)+'.build.log')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix='.nn-build-',dir=binary.parent) as tmp:
        env['TMPDIR']=tmp
        recorded_env=record_environment(env)
        with log.open('w') as f:
            subprocess.run(command,cwd=snapshot,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        # Retain generated Cargo configuration where available, without copying its target cache.
        generated_cargo=collect_generated_cargo(tmp,snapshot)
    for relative,digest in inputs.items():require(sha(snapshot/relative)==digest,'frozen source changed during build')
    data=dict(schema='nn-cpu-build-v1',status='complete',started_utc=started,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        origin_repository=str(REPO),cwd=str(snapshot),snapshot_directory=snapshot.name,
        binary=str(binary),binary_sha256=sha(binary),candidate_source_sha256=inputs['native/cpu_fast.rs'],
        harness_sha256=inputs['examples/_duel_prefill.almd'],config_sha256=inputs['almide.toml'],
        inputs=inputs,generated_cargo=generated_cargo,compiler=compiler,rustc=rustc,cargo=cargo,command=command,environment=recorded_env,
        build_log_sha256=sha(log),builder_sha256=builder_sha)
    sidecar.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(binary=str(binary),sidecar=str(sidecar),binary_sha256=data['binary_sha256'],
        candidate_source_sha256=data['candidate_source_sha256']),indent=2))

if __name__=='__main__':main()
