"""Kaggle kernel for v231 (clamp power for a faint value; docs/prereg_v231_clamp_power.md). Build only.
astrocyte-v231-power (GPU T4 x2: gpu0 seeds 2, 0; gpu1 seeds 1, 3, 4; each model retrained and checked against v230's
sha256, then the dose grid). Push with --accelerator NvidiaTeslaT4. Account from V231_ACCOUNT (default aurshalevmerin).
Child output is streamed to the console and saved per process.
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); ACCT = os.environ.get('V231_ACCOUNT', 'aurshalevmerin')
JOBS = [('gpu0', '0', '2,0'), ('gpu1', '1', '1,3,4')]
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in ('v220_train.py', 'v230_pilot.py', 'v230_protocol.py', 'v178_analyze.py', 'v231_power.py')}
body = r'''import subprocess, sys, os, base64, zlib, hashlib, threading
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    procs = []
    for name, gpu, seeds in JOBS:
        os.makedirs(name, exist_ok=True)
        env = dict(os.environ, OUT_DIR='/kaggle/working/' + name, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES=gpu)
        procs.append(subprocess.Popen([sys.executable, '-u', 'v231_power.py', 'run', seeds], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True))
    def pump(name, p):
        with open(name + '/stdout.txt', 'w') as fh:
            for line in p.stdout:
                print('[%s] %s' % (name, line), end='', flush=True); fh.write(line); fh.flush()
    ts = [threading.Thread(target=pump, args=(j[0], p)) for j, p in zip(JOBS, procs)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print('exit codes', [p.wait() for p in procs], flush=True)
    for fn in FILES:
        os.remove(fn)
'''
dd = os.path.join(HERE, 'kaggle-kernel-v231-power'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nJOBS = %r\n' % (emb, JOBS) + body)
json.dump(dict(id='%s/astrocyte-v231-power' % ACCT, title='astrocyte-v231-power', code_file='run.py', language='python',
               kernel_type='script', is_private=True, enable_gpu=True, enable_internet=False, kernel_sources=[]),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, ACCT)
