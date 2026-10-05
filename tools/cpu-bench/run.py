#!/usr/bin/env python3
"""Fixed-work CPU comparison with a mandatory numerical gate and raw artifacts."""
import argparse, csv, fcntl, hashlib, json, math, os, pathlib, platform, statistics, subprocess, sys

from evidence import validate_build, validate_quality, validate_nn_result

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_SHA = '3d1cfc4b845efcda96d63415c986d51781d02b2f55f93fc6add142cd56b4e314'
LLAMA_SHA = '8e1642198dcd4e408f8776222d6ae31b74d01187'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=pathlib.Path)
    p.add_argument('--nn-bin', required=True, type=pathlib.Path)
    p.add_argument('--llama-bin', required=True, type=pathlib.Path)
    p.add_argument('--nn-source', type=pathlib.Path, help='optional extra source hash cross-check; cannot replace build sidecar')
    p.add_argument('--nn-build', type=pathlib.Path, help='defaults to <nn-bin>.build.json')
    p.add_argument('--quality-summary', required=True, type=pathlib.Path)
    p.add_argument('--out', required=True, type=pathlib.Path)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--repetitions', type=int, default=10)
    a = p.parse_args()
    if sys.platform != 'linux' or not hasattr(os, 'sched_getaffinity'):
        p.error('This measured protocol requires Linux affinity and KiB RSS accounting')
    available = sorted(os.sched_getaffinity(0))
    if not 1 <= a.threads <= len(available):
        p.error('Requested threads exceed available CPU affinity')
    if a.repetitions < 1:
        p.error('Repetitions must be positive')
    for name in ['model', 'nn_bin', 'llama_bin', 'quality_summary', 'out']:
        setattr(a, name, getattr(a, name).resolve())
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Use a fresh output directory; existing evidence is never overwritten')
    a.out.mkdir(parents=True, exist_ok=True)
    # Prevent accidentally running two copies of this protocol in one checkout.
    lock = open(HERE / '.benchmark.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    model_sha = sha(a.model)
    a.nn_build=(a.nn_build or pathlib.Path(str(a.nn_bin)+'.build.json')).resolve()
    try:
        build=validate_build(a.nn_build,a.nn_bin)
        source_sha=build['candidate_source_sha256']
        if a.nn_source and sha(a.nn_source)!=source_sha:p.error('--nn-source differs from the attested source')
    except (ValueError,KeyError,OSError) as e:p.error(f'Invalid nn build evidence: {e}')
    if model_sha != MODEL_SHA:
        p.error('Model SHA256 differs from the fixed comparison model')
    try:
        gate=validate_quality(a.quality_summary,model_sha,source_sha,a.threads,a.repetitions>=10)
    except (ValueError,KeyError,OSError) as e:p.error(f'Invalid numerical evidence: {e}')
    fixtures = json.loads((HERE.parent / 'cpu-parity/prompts.json').read_text())
    selected = available[:a.threads]
    cpus = ','.join(map(str, selected))
    env = os.environ.copy()
    env.update(ALMIDE_LOCKSTEP_THREADS=str(a.threads), RAYON_NUM_THREADS=str(a.threads))
    # Diagnostic overrides must not silently carry into a published default comparison.
    for name in ['NN_PREFILL_PROFILE', 'NN_PREFILL_KERNEL', 'NN_PREFILL_ATTENTION']:
        env.pop(name, None)
    manifest = {
        'model_sha256': model_sha, 'candidate_source_sha256': source_sha,
        'nn_build_sidecar_sha256':sha(a.nn_build),'nn_build_sidecar':str(a.nn_build),
        'nn_harness_sha256':build['harness_sha256'],'nn_binary_sha256': sha(a.nn_bin), 'llama_binary_sha256': sha(a.llama_bin),
        'llama_source_sha': LLAMA_SHA, 'quality_summary_sha256': sha(a.quality_summary),
        'quality_full_suite': gate.get('full_suite'), 'threads': a.threads, 'affinity': selected,
        'repetitions': a.repetitions, 'platform': platform.platform(),
        'cpuinfo': pathlib.Path('/proc/cpuinfo').read_text(),
        'workload': 'input128/1024; context2048; 128 timed additional decode calls and argmax; 129 emitted IDs; full prompt+8 warmup; no EOS exit',
        'scope': 'one CPU host/model/quantization; not a universal engine or quality claim',
    }
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    engines = ['nn', 'llama-f32', 'llama-f16']
    records, design = [], []
    for n in [128, 1024]:
        tokens = fixtures[f'input_{n}']
        assert len(tokens) == n
        tokenfile = a.out / f'input-{n}.tokens'
        tokenfile.write_text(' '.join(map(str, tokens)) + '\n')
        counts = dict.fromkeys(engines, 0)
        orders = []
        if a.repetitions % 2 == 0:
            for block in range(a.repetitions // 2):
                order = engines[block % 3:] + engines[:block % 3]
                orders.append((block, order + order[::-1]))
        else:
            for block in range(a.repetitions):
                orders.append((block, engines[block % 3:] + engines[:block % 3]))
        for block, order in orders:
            for engine in order:
                rep = counts[engine]
                counts[engine] += 1
                prefix = a.out / f'p{n}-{engine}-{rep:02}'
                cmd = [a.nn_bin if engine == 'nn' else a.llama_bin, a.model, tokenfile, 128, a.threads, prefix]
                if engine != 'nn':
                    cmd += ['--kv', engine.split('-')[1], '--ctx', 2048]
                spec = {'input': n, 'engine': engine, 'rep': rep, 'block': block, 'command': list(map(str, cmd))}
                design.append(spec)
                (a.out / 'design.json').write_text(json.dumps(design, indent=2))
                wrapped = [sys.executable, HERE / 'measure_one.py', '--prefix', prefix, '--', 'taskset', '-c', cpus] + cmd
                subprocess.run(list(map(str, wrapped)), env=env, stdout=subprocess.DEVNULL, check=True)
                m = json.loads(pathlib.Path(str(prefix) + '.json').read_text())
                proc = json.loads(pathlib.Path(str(prefix) + '.process.json').read_text())
                if engine == 'nn':
                    validate_nn_result(m,n,a.threads)
                    prefill, decode = m['prefill_ms'] / 1000, m['decode_ms'] / 1000
                    latency = m['latencies_ms']
                else:
                    assert m['source_sha'] == LLAMA_SHA and m['prompt_tokens'] == n and m['decode_steps'] == 128
                    assert m['emitted_ids'] == 129 and m['kv_type'] == engine.split('-')[1]
                    assert m['context_actual'] == 2048 and not m['correctness_mode']
                    prefill, decode = m['prefill_s'] + m['first_argmax_s'], m['generation_wall_s']
                    latency = m['latency_ms']
                assert m['threads'] == a.threads and len(latency) == 128
                assert all(math.isfinite(v) and v >= 0 for v in latency)
                assert all(math.isfinite(v) and v > 0 for v in [prefill, decode])
                row = {**{k: spec[k] for k in ['input', 'engine', 'rep', 'block']},
                       'prefill_s': prefill, 'decode_s': decode, 'total_s': prefill + decode,
                       'decode_tokens_per_s': 128 / decode, 'peak_rss_mib': proc['peak_rss_kib'] / 1024,
                       'latency_median_ms': statistics.median(latency),
                       'latency_p95_ms': sorted(latency)[math.ceil(.95 * len(latency)) - 1]}
                records.append(row)
                with (a.out / 'trials.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(records)
                print(json.dumps(row), flush=True)
    summary = []
    for n in [128, 1024]:
        for engine in engines:
            rows = [r for r in records if r['input'] == n and r['engine'] == engine]
            assert len(rows) == a.repetitions
            summary.append({'input': n, 'engine': engine, 'repetitions': len(rows), **{
                metric: {'median': statistics.median(r[metric] for r in rows),
                         'min': min(r[metric] for r in rows), 'max': max(r[metric] for r in rows)}
                for metric in ['prefill_s', 'decode_s', 'total_s', 'decode_tokens_per_s', 'peak_rss_mib', 'latency_median_ms', 'latency_p95_ms']}})
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
