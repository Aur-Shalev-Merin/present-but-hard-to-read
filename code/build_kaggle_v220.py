"""Kaggle GPU kernel for v220 (docs/v220_design.md; study: docs/prereg_v220_training_order.md). Build only.
  V220_SLUG=study: astrocyte-v220-study, the 18 study cells (v220_study.py) alternated over the two GPUs.
  astrocyte-v220-<V220_SLUG> on account aurmerin (V220_ACCOUNT), T4 x2: the pilot cells (V220_ARCHS, default std,loop) split over the two GPUs, run in
  parallel, child output streamed to the console (and saved per GPU).
Push with --accelerator NvidiaTeslaT4.
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); ACCT = os.environ.get('V220_ACCOUNT', 'aurmerin')
ARCHS = os.environ.get('V220_ARCHS', 'std,loop').split(','); SLUG = os.environ.get('V220_SLUG', 'pilot')
STUDY = SLUG == 'study'
if STUDY:   # the 18 study cells (v220_study.GRID order, dig first), alternated over the GPUs
    cells = ['%s:%s:%d' % (a, f, s) for f in ('dig', 'mod') for a in ('std', 'loop', 'std2') for s in (1, 2, 3)]
else:
    cells = ['%s:%s:%s:0' % (a, f, lr) for a in ARCHS for f in ('dig', 'mod') for lr in ('3e-4', '1e-3', '3e-3')]
SCRIPT = 'v220_study.py' if STUDY else 'v220_train.py'
split = [','.join(cells[0::2]), ','.join(cells[1::2])]
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in (['v220_train.py', 'v220_measure.py', 'v220_study.py'] if STUDY else ['v220_train.py'])}
body = r'''import subprocess, sys, os, base64, zlib, hashlib, threading
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    procs = []
    for gpu, cells in enumerate(SPLIT):
        os.makedirs('gpu%d' % gpu, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), V220_CELLS=cells, OUT_DIR='/kaggle/working/gpu%d' % gpu)
        procs.append(subprocess.Popen([sys.executable, '-u', SCRIPT], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True))
    def pump(i, p):
        with open('gpu%d/stdout.txt' % i, 'w') as fh:
            for line in p.stdout:
                print('[gpu%d] %s' % (i, line), end='', flush=True); fh.write(line); fh.flush()
    ts = [threading.Thread(target=pump, args=(i, p)) for i, p in enumerate(procs)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print('exit codes', [p.wait() for p in procs], flush=True)
    for fn in FILES:
        os.remove(fn)
'''
dd = os.path.join(HERE, 'kaggle-kernel-v220-' + SLUG); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nSPLIT = %r\nSCRIPT = %r\n' % (emb, split, SCRIPT) + body)
json.dump(dict(id='%s/astrocyte-v220-%s' % (ACCT, SLUG), title='astrocyte-v220-' + SLUG, code_file='run.py', language='python',
               kernel_type='script', is_private=True, enable_gpu=True, enable_internet=False, kernel_sources=[]),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, ACCT)
