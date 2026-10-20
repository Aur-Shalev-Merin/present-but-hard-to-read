"""Kaggle CPU kernels for v212 (docs/prereg_v212_mixture_null.md). Build only.
  V212_MODE=selftest      astrocyte-v212-selftest : v212_mixture_null.py selftest
  V212_MODE=cal_<family>  astrocyte-v212-cal-<f>  : stage 1, calibration (seed 100, LAMS_CAL or V212_LAMS, needed only)
  V212_MODE=run_<family>  astrocyte-v212-run-<f>  : stage 2, seeds SEEDS at V212_LAM (lam* from `match` on stage 1)
BLAS threads are pinned to 1 per worker process (4 workers on Kaggle's 4 CPUs).
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V212_MODE', 'selftest')
FILES = ['v212_mixture_null.py', 'v194_analyze.py', 'v186_analyze.py', 'v178_analyze.py', 'v194_items.json']
import v212_mixture_null as M
if MODE == 'selftest':
    steps = [['selftest']]
elif MODE.startswith('cal_'):
    steps = [['calib', MODE[4:]] + os.environ.get('V212_LAMS', '').split()]   # amendment 1: grid x16 (prereg section 2)
else:
    steps = [['run', MODE[4:], os.environ['V212_LAM']] + [str(s) for s in M.SEEDS]]
emb = {}
for f in FILES:
    raw = open(os.path.join(HERE, f), 'rb').read()
    emb[f] = base64.b64encode(zlib.compress(raw.replace(b'\r\n', b'\n') if f.endswith('.py') else raw, 9)).decode()
body = r'''import subprocess, sys, os, base64, zlib, hashlib
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    env = dict(os.environ, OUT_DIR='/kaggle/working', PYTHONWARNINGS='ignore', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    rcs = []
    with open('v212_stdout.txt', 'w') as fh:
        for args in STEPS:
            rcs.append(subprocess.call([sys.executable, '-u', 'v212_mixture_null.py'] + args, stdout=fh,
                                       stderr=subprocess.STDOUT, env=env))
            fh.flush()
    print(open('v212_stdout.txt').read()[-8000:], flush=True)
    for fn in FILES:
        os.remove(fn)
    if any(rcs):
        raise SystemExit('a step failed: %s' % rcs)
'''
kid = 'astrocyte-v212-' + MODE.replace('_', '-')
dd = os.path.join(HERE, 'kaggle-kernel-v212-' + MODE.replace('_', '-')); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nSTEPS = %r\n' % (emb, steps) + body)
json.dump(dict(id='aurshalevmerin/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=[]),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, steps)
