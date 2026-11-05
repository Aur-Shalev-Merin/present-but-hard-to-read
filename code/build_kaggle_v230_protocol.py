"""Kaggle kernels for the v230 protocol arm (docs/prereg_v230_positive_control.md). Build only.
MODE=nullbar  astrocyte-v230-nullbar  (CPU; pre-freeze: null worlds seeds 1-6 -> TAU230)
MODE=main     astrocyte-v230-protocol (GPU T4 x2: gpu0 pos models seeds 0-4, gpu1 neg models seeds 0-4; push with
              --accelerator NvidiaTeslaT4)
MODE=null     astrocyte-v230-nullfresh (CPU; fresh null seeds 7-12)
Account from V230_ACCOUNT (default aurshalevmerin). Child output is streamed to the console and saved per process.
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); ACCT = os.environ.get('V230_ACCOUNT', 'aurshalevmerin')
MODE = os.environ.get('MODE', 'main')
SLUG = {'nullbar': 'nullbar', 'main': 'protocol', 'null': 'nullfresh'}[MODE]
JOBS = {'nullbar': [('cpu', None, '--nullbar')], 'main': [('gpu0', '0', 'pos'), ('gpu1', '1', 'neg')], 'null': [('cpu', None, 'null')]}[MODE]
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in ('v220_train.py', 'v230_pilot.py', 'v230_protocol.py', 'v178_analyze.py')}
body = r'''import subprocess, sys, os, base64, zlib, hashlib, threading
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    procs = []
    for name, gpu, arg in JOBS:
        os.makedirs(name, exist_ok=True)
        env = dict(os.environ, OUT_DIR='/kaggle/working/' + name, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = gpu
        procs.append(subprocess.Popen([sys.executable, '-u', 'v230_protocol.py', arg], env=env, stdout=subprocess.PIPE,
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
dd = os.path.join(HERE, 'kaggle-kernel-v230-' + SLUG); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nJOBS = %r\n' % (emb, JOBS) + body)
json.dump(dict(id='%s/astrocyte-v230-%s' % (ACCT, SLUG), title='astrocyte-v230-' + SLUG, code_file='run.py', language='python',
               kernel_type='script', is_private=True, enable_gpu=MODE == 'main', enable_internet=False, kernel_sources=[]),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, ACCT, MODE)
