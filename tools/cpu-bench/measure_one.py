#!/usr/bin/env python3
"""Measure one isolated benchmark process. Wall includes load/warmup; phase timing is internal."""
import argparse, datetime, json, os, pathlib, resource, subprocess, time
p=argparse.ArgumentParser()
p.add_argument('--prefix', required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a=p.parse_args()
cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
prefix=pathlib.Path(a.prefix)
prefix.parent.mkdir(parents=True,exist_ok=True)
t=time.perf_counter()
with open(str(prefix)+'.stdout.log','w') as out, open(str(prefix)+'.stderr.log','w') as err:
    cp=subprocess.run(cmd,stdout=out,stderr=err)
wall=time.perf_counter()-t
r=resource.getrusage(resource.RUSAGE_CHILDREN)
data={'command':cmd,'returncode':cp.returncode,'wall_s':wall,'peak_rss_kib':r.ru_maxrss,
 'user_cpu_s':r.ru_utime,'system_cpu_s':r.ru_stime,'minor_faults':r.ru_minflt,'major_faults':r.ru_majflt,
 'voluntary_context_switches':r.ru_nvcsw,'involuntary_context_switches':r.ru_nivcsw,
 'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ALMIDE_LOCKSTEP_THREADS':os.environ.get('ALMIDE_LOCKSTEP_THREADS'),'RAYON_NUM_THREADS':os.environ.get('RAYON_NUM_THREADS')}
pathlib.Path(str(prefix)+'.process.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data),flush=True)
raise SystemExit(cp.returncode)
