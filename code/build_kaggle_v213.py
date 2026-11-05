"""Kaggle kernels for v213 (docs/prereg_v213_scale_up.md). Build only.
  V213_MODE=pilot       astrocyte-v213-pilot    (T4 x2): v213_pilot.py, qwen7b then ouro26 (timing/correctness; done)
  V213_MODE=selftest    astrocyte-v213-selftest (CPU):   v213_analyze.py --selftest
  V213_MODE=check       astrocyte-v213-check    (CPU):   selftest, then v213_items.py (prints the items hash on Kaggle)
  V213_MODE=smoke       astrocyte-v213-smoke    (T4 x2): v213_extract.py with V213_PILOT=1, every part, both models
  V213_MODE=x-qwen7b    astrocyte-v213-x-qwen7b (T4 x2): parts v181, b1, b3, b2 (one shard), Qwen2.5-7B over both GPUs
  V213_MODE=x-ouro26    astrocyte-v213-x-ouro26 (T4 x2): GPU 0: v181, b1, b3, b2 shard 0/2; GPU 1: b2 shard 1/2
  V213_MODE=a-<model>-<parts>  astrocyte-v213-a-<model>-<parts> (CPU): v213_analyze.py on x-<model>'s output, e.g.
                         a-qwen7b-AB1B3 or a-ouro26-B2
  V213_MODE=s-<model>-<K>      astrocyte-v213-s-<model>-<K> (CPU): v213_split.py part K (B2x, layer K only)
  V213_MODE=c-<model>          astrocyte-v213-c-<model> (CPU): v213_split.py combine (B2x report from parts 0-3)
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get('V213_MODE', 'smoke')
EXTRACT = ['v213_extract.py', 'v213_items.py', 'v202_extract.py', 'v205_extract.py', 'v178_labels.json']
ANALYZE = ['v213_analyze.py', 'v202_analyze.py', 'v205_analyze.py', 'v205_saved.py', 'v178_analyze.py', 'v181_swamp_check.py']
gpu, internet, sources, datasets = True, True, [], []
if MODE == 'pilot':
    files = ['v213_pilot.py']
    procs = [[('v213_pilot.py', [], dict(V213_MODEL='qwen7b'), '0,1')], [('v213_pilot.py', [], dict(V213_MODEL='ouro26'), '0')]]
elif MODE == 'selftest':
    files, gpu, internet = ANALYZE, False, False
    procs = [[('v213_analyze.py', ['--selftest'], {}, '')]]
elif MODE == 'check':                                     # selftest + rebuild the items on Kaggle (hash check)
    files, gpu = ANALYZE + ['v213_items.py', 'v205_extract.py', 'v202_extract.py'], False
    procs = [[('v213_analyze.py', ['--selftest'], {}, ''), ('v213_items.py', [], {}, '')]]
elif MODE == 'smoke':
    files = EXTRACT
    procs = [[('v213_extract.py', [], dict(V213_MODEL='qwen7b', V213_PART=p, V213_PILOT='1'), '0,1') for p in ('v181', 'b1', 'b2', 'b3')]
             + [('v213_extract.py', [], dict(V213_MODEL='ouro26', V213_PART=p, V213_PILOT='1'), '0') for p in ('v181', 'b1', 'b2', 'b3')]]
elif MODE == 'x-qwen7b':
    files = EXTRACT
    procs = [[('v213_extract.py', [], dict(V213_MODEL='qwen7b', V213_PART=p, V213_SHARD='0', V213_NSHARD='1'), '0,1')
              for p in ('v181', 'b1', 'b3', 'b2')]]
elif MODE == 'x-ouro26':
    files = EXTRACT
    procs = [[('v213_extract.py', [], dict(V213_MODEL='ouro26', V213_PART=p, V213_SHARD='0', V213_NSHARD='2'), '0')
              for p in ('v181', 'b1', 'b3', 'b2')],
             [('v213_extract.py', [], dict(V213_MODEL='ouro26', V213_PART='b2', V213_SHARD='1', V213_NSHARD='2'), '1')]]
elif MODE.startswith('s-'):                               # s-<model>-<K>: B2x part K (v213_split.py)
    _, model, k = MODE.split('-')
    files, gpu, internet = ANALYZE + ['v213_split.py'], False, False
    sources = ['aurshalevmerin/astrocyte-v213-x-%s' % model]
    procs = [[('v213_split.py', ['/kaggle/input', model, 'part', k], {}, '')]]
elif MODE.startswith('r-'):                               # r-<model>-<K>: part K's unfinished jobs, 4 BLAS threads
    _, model, k = MODE.split('-')
    files, gpu, internet = ANALYZE + ['v213_split.py'], False, False
    sources = ['aurshalevmerin/astrocyte-v213-x-%s' % model] + (
        [] if os.environ.get('V213_NOPREV') else ['aurshalevmerin/astrocyte-v213-s-%s-%s' % (model, k)])   # NOPREV: all jobs
    datasets = [os.environ['V213_SAVED_DS']] if os.environ.get('V213_SAVED_DS') else []   # saved results as a dataset
    procs = [[('v213_split.py', ['/kaggle/input', model, 'rest', k],
               dict(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'), '')]]
elif MODE.startswith('c-'):                               # c-<model>[-r]: B2x combine of parts 0-3 (-r: from r- kernels)
    model, rest = MODE[2:].split('-')[0], MODE.endswith('-r')
    files, gpu, internet = ANALYZE + ['v213_split.py'], False, False
    sources = ['aurshalevmerin/astrocyte-v213-x-%s' % model] + ['aurshalevmerin/astrocyte-v213-%s-%s-%d' % (
        'r' if rest else 's', model, k) for k in range(4)]
    procs = [[('v213_split.py', ['/kaggle/input', model, 'combine'], {}, '')]]
else:                                                     # a-<model>-<parts>
    _, model, parts = MODE.split('-', 2)
    plist = [p for p in ('A', 'B1', 'B2', 'B3') if p in parts.replace('B1', '|B1|').replace('B2', '|B2|').replace('B3', '|B3|').split('|')]
    files, gpu, internet = ANALYZE, False, False
    sources = ['aurshalevmerin/astrocyte-v213-x-%s' % model]
    procs = [[('v213_analyze.py', ['/kaggle/input', model] + plist, {}, '')]]
emb = {f: base64.b64encode(zlib.compress(
    open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n') if f.endswith('.py') else open(os.path.join(HERE, f), 'rb').read(), 9)).decode()
    for f in files}
body = r'''import subprocess, sys, os, base64, zlib, hashlib, time
if __name__ == '__main__':
    W = '/kaggle/working'; os.chdir(W)
    for fn, e in FILES.items():
        b = zlib.decompress(base64.b64decode(e)); open(fn, 'wb').write(b); print(fn, hashlib.sha256(b).hexdigest()[:16], flush=True)
    if INTERNET:
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets', 'accelerate'])
    base = dict(os.environ, OUT_DIR=W, PYTHONUNBUFFERED='1', PYTHONWARNINGS='ignore', HF_HUB_DISABLE_PROGRESS_BARS='1',
                PYTORCH_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def chain(steps, tag):
        rcs = []
        with open('v213_stdout_%s.txt' % tag, 'w') as fh:
            for script, args, env, gpus in steps:
                e = dict(base, **env)
                if gpus:
                    e['CUDA_VISIBLE_DEVICES'] = gpus
                rcs.append(subprocess.call([sys.executable, '-u', script] + args, stdout=fh, stderr=subprocess.STDOUT, env=e))
                fh.flush()
                if rcs[-1]:
                    break
        return rcs
    import threading
    out = {}
    ths = [threading.Thread(target=lambda i=i, s=s: out.__setitem__(i, chain(s, str(i)))) for i, s in enumerate(PROCS)]
    for t in ths:
        t.start(); time.sleep(30)
    for t in ths:
        t.join()
    for i in range(len(PROCS)):
        print('--- process %d exits %s' % (i, out.get(i)), flush=True); print(open('v213_stdout_%d.txt' % i).read()[-6000:], flush=True)
    for fn in FILES:
        os.remove(fn)
    if any(any(r) for r in out.values()):
        raise SystemExit('a step failed: %s' % out)
'''
kid = 'astrocyte-v213-' + MODE
dd = os.path.join(HERE, 'kaggle-kernel-v213-' + MODE); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    'FILES = %r\nPROCS = %r\nINTERNET = %r\n' % (emb, procs, internet) + body)
json.dump(dict(id='aurshalevmerin/' + kid, title=kid, code_file='run.py', language='python', kernel_type='script',
               is_private=True, enable_gpu=gpu, enable_internet=internet, kernel_sources=sources, dataset_sources=datasets),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, 'gpu' if gpu else 'cpu', [len(p) for p in procs])
