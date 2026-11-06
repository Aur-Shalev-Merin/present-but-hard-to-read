"""Kaggle kernels for v233 (clamp power for the Kudo form; docs/prereg_v233_kudo_power.md). Build only. CPU, no internet;
kernel_sources aurshalevmerin/astrocyte-v231-power (the five v230 pos checkpoints, sha-checked in the script).

  V233_MODE=known  -> astrocyte-v233-known : known answers on the seed-0 checkpoint (pre-freeze)
  V233_MODE=run    -> astrocyte-v233-power : the grid, two processes (seeds 0,2 and 1,3,4), then the verdict
Child output is streamed to the console and saved per process.
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); MODE = os.environ.get('V233_MODE', 'known')
FILES = ('v220_train.py', 'v230_pilot.py', 'v230_protocol.py', 'v178_analyze.py', 'v231_power.py', 'v233_kudo_power.py')
emb = {}
for f in FILES:
    b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode(); print('  %s sha256 %s' % (f, hashlib.sha256(b).hexdigest()[:16]))
JOBS = [('known', ['known', '/kaggle/input'])] if MODE == 'known' else \
       [('p0', ['run', '0,2', '/kaggle/input']), ('p1', ['run', '1,3,4', '/kaggle/input'])]
body = r'''import subprocess, sys, os, base64, zlib, hashlib, threading
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    procs = []
    for name, args in JOBS:
        os.makedirs(name, exist_ok=True)
        env = dict(os.environ, OUT_DIR='/kaggle/working/' + name, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        procs.append(subprocess.Popen([sys.executable, '-u', 'v233_kudo_power.py'] + args, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True))
    def pump(name, p):
        with open(name + '/stdout.txt', 'w') as fh:
            for line in p.stdout:
                print('[%s] %s' % (name, line), end='', flush=True); fh.write(line); fh.flush()
    ts = [threading.Thread(target=pump, args=(j[0], p)) for j, p in zip(JOBS, procs)]
    [t.start() for t in ts]; [t.join() for t in ts]
    rcs = [p.wait() for p in procs]; print('exit codes', rcs, flush=True)
    if MODE == 'run' and not any(rcs):
        with open('v233_verdict.txt', 'w') as fh:
            r = subprocess.run([sys.executable, '-u', 'v233_kudo_power.py', 'verdict', '/kaggle/working'], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
            fh.write(r.stdout); print(r.stdout, flush=True)
    for fn in FILES:
        os.remove(fn)
'''
kid = 'astrocyte-v233-' + ('known' if MODE == 'known' else 'power')
dd = os.path.join(HERE, 'kaggle-kernel-v233-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nJOBS = %r\nMODE = %r\n' % (emb, JOBS, MODE) + body)
json.dump(dict(id='aurshalevmerin/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=False, enable_internet=False, kernel_sources=['aurshalevmerin/astrocyte-v231-power']),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
